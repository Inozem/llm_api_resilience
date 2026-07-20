"""Public resilient LLM facade."""

from datetime import datetime, timezone
from time import perf_counter, sleep
from typing import Any, Dict, Optional, Tuple

from .attempts import AttemptRecord
from .classifiers import DefaultFailureClassifier, FailureClassifier
from .circuit_breaker import CircuitState
from .errors import CircuitOpenError, FailoverExhaustedError
from .observability import CircuitEvent
from .responses import ResilientChatResponse
from .result_policies import normalize_result_policy
from .routes import RecoveryPlan, Route


class ResilientLLM:
    """Execute chat requests with retry and ordered route failover."""

    def __init__(
        self,
        recovery_plan: RecoveryPlan,
        failure_classifier: Optional[FailureClassifier] = None,
        result_policy: Any = None,
    ):
        if not isinstance(recovery_plan, RecoveryPlan):
            raise TypeError("recovery_plan must be a RecoveryPlan")
        if failure_classifier is None:
            failure_classifier = DefaultFailureClassifier()
        if not isinstance(failure_classifier, FailureClassifier):
            raise TypeError(
                "failure_classifier must provide an is_retryable method"
            )
        self._recovery_plan = recovery_plan
        self._failure_classifier = failure_classifier
        self._result_policy = normalize_result_policy(result_policy)
        self._last_attempts: Tuple[AttemptRecord, ...] = ()
        self._last_events: Tuple[CircuitEvent, ...] = ()

    @property
    def recovery_plan(self) -> RecoveryPlan:
        """The immutable plan used by this facade."""

        return self._recovery_plan

    @property
    def last_attempts(self) -> Tuple[AttemptRecord, ...]:
        """Metadata for the most recent call, including failed calls."""

        return self._last_attempts

    @property
    def last_events(self) -> Tuple[CircuitEvent, ...]:
        """Safe circuit-breaker events from the most recent operation."""

        return self._last_events

    @property
    def failure_classifier(self) -> FailureClassifier:
        """Classifier used for retry and failover decisions."""

        return self._failure_classifier

    @property
    def result_policy(self) -> Any:
        """Optional policy configured for result validation."""

        return self._result_policy

    def session(self, messages: Any, *, journal: Any = None, **kwargs: Any):
        """Create an application-managed session for tool-calling recovery."""

        from .session import ResilientSession

        return ResilientSession(self, messages, kwargs, journal=journal)

    def chat(self, messages: Any, **kwargs: Any) -> ResilientChatResponse:
        """Retry transient failures and fail over through the route order."""

        attempts = []
        last_error: Optional[Exception] = None
        blocked_cooldowns = []
        self._last_attempts = ()
        events = []
        self._last_events = ()

        for route_index, route in enumerate(self._recovery_plan):
            if not self._allow_route_request(route, events):
                if route.breaker is not None:
                    blocked_cooldowns.append(
                        route.breaker.snapshot().cooldown_remaining_s
                    )
                continue

            for failed_attempt in range(1, route.policy.max_attempts + 1):
                request_kwargs = self._build_request_kwargs(
                    kwargs,
                    route=route,
                    include_previous_response=route_index == 0,
                )
                request_messages = self._build_request_messages(
                    messages,
                    route=route,
                )
                started_at = datetime.now(timezone.utc)
                started_tick = perf_counter()

                try:
                    response = route.adapter.chat(
                        messages=request_messages,
                        **request_kwargs,
                    )
                except Exception as error:
                    duration_s = perf_counter() - started_tick
                    attempt = self._make_attempt_record(
                        route=route,
                        started_at=started_at,
                        duration_s=duration_s,
                        success=False,
                        error=error,
                    )
                    attempts.append(attempt)
                    self._last_attempts = tuple(attempts)
                    last_error = error
                    is_retryable = self._failure_classifier.is_retryable(error)
                    if is_retryable:
                        self._record_route_failure(route, error, events)

                    if not is_retryable:
                        raise

                    if (
                        failed_attempt >= route.policy.max_attempts
                        or (
                            route.breaker is not None
                            and route.breaker.state is CircuitState.OPEN
                        )
                    ):
                        break

                    delay_s = route.policy.backoff_for(failed_attempt)
                    if delay_s > 0:
                        sleep(delay_s)
                    continue

                duration_s = perf_counter() - started_tick
                attempt = self._make_attempt_record(
                    route=route,
                    started_at=started_at,
                    duration_s=duration_s,
                    success=True,
                )
                attempts.append(attempt)
                self._last_attempts = tuple(attempts)
                self._record_route_success(route, events)
                return ResilientChatResponse.from_chat_response(
                    response,
                    selected_route=route.name,
                    attempts=tuple(attempts),
                    events=tuple(events),
                )

        if last_error is not None:
            raise FailoverExhaustedError(attempts, last_error) from last_error
        if blocked_cooldowns:
            raise CircuitOpenError(cooldown_remaining_s=min(blocked_cooldowns))
        raise RuntimeError("recovery plan did not execute any route")

    def _allow_route_request(self, route: Route, events: list) -> bool:
        breaker = route.breaker
        if breaker is None:
            return True

        previous_state = breaker.state
        allowed = breaker.allow_request()
        current_state = breaker.state

        if (
            allowed
            and previous_state is CircuitState.OPEN
            and current_state is CircuitState.HALF_OPEN
        ):
            events.append(
                self._make_circuit_event(
                    route,
                    event_type="half_open",
                    state=current_state,
                )
            )
        elif not allowed:
            events.append(
                self._make_circuit_event(
                    route,
                    event_type="skipped",
                    state=current_state,
                    cooldown_remaining_s=breaker.snapshot().cooldown_remaining_s,
                )
            )

        self._last_events = tuple(events)
        return allowed

    def _record_route_failure(
        self,
        route: Route,
        error: Exception,
        events: list,
    ) -> None:
        breaker = route.breaker
        if breaker is None:
            return

        previous_state = breaker.state
        breaker.record_failure()
        if (
            previous_state is not CircuitState.OPEN
            and breaker.state is CircuitState.OPEN
        ):
            events.append(
                self._make_circuit_event(
                    route,
                    event_type="opened",
                    state=breaker.state,
                    error=error,
                    cooldown_remaining_s=breaker.snapshot().cooldown_remaining_s,
                )
            )
        self._last_events = tuple(events)

    def _record_route_success(self, route: Route, events: list) -> None:
        breaker = route.breaker
        if breaker is None:
            return

        previous_state = breaker.state
        breaker.record_success()
        if previous_state is not CircuitState.CLOSED:
            events.append(
                self._make_circuit_event(
                    route,
                    event_type="closed",
                    state=breaker.state,
                )
            )
        self._last_events = tuple(events)

    def _make_circuit_event(
        self,
        route: Route,
        *,
        event_type: str,
        state: CircuitState,
        error: Optional[Exception] = None,
        cooldown_remaining_s: float = 0.0,
    ) -> CircuitEvent:
        adapter = route.adapter
        provider = self._get_adapter_string(adapter, "organization")
        if provider is None:
            provider = self._get_adapter_string(adapter, "company")
        model = self._get_adapter_string(adapter, "model")
        return CircuitEvent(
            event_type=event_type,
            route_name=route.name,
            state=state,
            provider=provider,
            model=model,
            error_type=type(error).__name__ if error is not None else None,
            cooldown_remaining_s=cooldown_remaining_s,
        )

    @staticmethod
    def _build_request_kwargs(
        original_kwargs: Dict[str, Any],
        *,
        route: Route,
        include_previous_response: bool,
    ) -> Dict[str, Any]:
        """Build isolated kwargs for one attempt without mutating the caller."""

        request_kwargs = dict(original_kwargs)
        if not include_previous_response:
            request_kwargs.pop("previous_response", None)
        if "timeout_s" not in request_kwargs and route.policy.timeout_s is not None:
            request_kwargs["timeout_s"] = route.policy.timeout_s
        return request_kwargs

    @staticmethod
    def _build_request_messages(messages: Any, *, route: Route) -> Any:
        """Build isolated route messages when a prompt profile is configured."""

        if route.prompt_profile is None:
            return messages
        return route.prompt_profile.apply_to_request(messages)

    def _make_attempt_record(
        self,
        *,
        route: Route,
        started_at: datetime,
        duration_s: float,
        success: bool,
        error: Optional[Exception] = None,
    ) -> AttemptRecord:
        adapter = route.adapter
        provider = self._get_adapter_string(adapter, "organization")
        if provider is None:
            provider = self._get_adapter_string(adapter, "company")
        model = self._get_adapter_string(adapter, "model")

        return AttemptRecord(
            route_name=route.name,
            provider=provider,
            model=model,
            started_at=started_at,
            duration_s=duration_s,
            success=success,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
        )

    @staticmethod
    def _get_adapter_string(adapter: Any, attribute: str) -> Optional[str]:
        value = getattr(adapter, attribute, None)
        return value if isinstance(value, str) and value else None
