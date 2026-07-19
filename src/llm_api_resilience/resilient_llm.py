"""Public resilient LLM facade."""

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

from .attempts import AttemptRecord
from .responses import ResilientChatResponse
from .routes import RecoveryPlan, Route


class ResilientLLM:
    """Execute chat requests through the first route in a recovery plan.

    Version 0.1 is intentionally a transparent single-route facade.  The
    route loop and retry behavior are introduced in v0.2.
    """

    def __init__(self, recovery_plan: RecoveryPlan):
        if not isinstance(recovery_plan, RecoveryPlan):
            raise TypeError("recovery_plan must be a RecoveryPlan")
        self._recovery_plan = recovery_plan
        self._last_attempts: Tuple[AttemptRecord, ...] = ()

    @property
    def recovery_plan(self) -> RecoveryPlan:
        """The immutable plan used by this facade."""

        return self._recovery_plan

    @property
    def last_attempts(self) -> Tuple[AttemptRecord, ...]:
        """Metadata for the most recent call, including failed calls."""

        return self._last_attempts

    def chat(self, messages: Any, **kwargs: Any) -> ResilientChatResponse:
        """Delegate one chat request to the first route in the plan."""

        route = self._recovery_plan[0]
        request_kwargs: Dict[str, Any] = dict(kwargs)
        if "timeout_s" not in request_kwargs and route.policy.timeout_s is not None:
            request_kwargs["timeout_s"] = route.policy.timeout_s

        started_at = datetime.now(timezone.utc)
        started_tick = perf_counter()

        try:
            response = route.adapter.chat(messages=messages, **request_kwargs)
            duration_s = perf_counter() - started_tick
            attempt = self._make_attempt_record(
                route=route,
                started_at=started_at,
                duration_s=duration_s,
                success=True,
            )
            resilient_response = ResilientChatResponse.from_chat_response(
                response,
                selected_route=route.name,
                attempts=(attempt,),
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
            self._last_attempts = (attempt,)
            raise

        self._last_attempts = (attempt,)
        return resilient_response

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
