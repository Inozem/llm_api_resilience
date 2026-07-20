import pytest

from llm_api_adapter.errors import (
    JSONSchemaError,
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from llm_api_adapter.errors.config_errors import LLMConfigError
from llm_api_adapter.errors.llm_api_error import InvalidToolSchemaError
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools import ToolCall

from llm_api_resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    RecoveryPlan,
    FailoverExhaustedError,
    ResilientChatResponse,
    ResilientLLM,
    Route,
    RoutePolicy,
)

pytestmark = pytest.mark.unit


class FakeAdapter:
    def __init__(self, *, provider="fake", model="fake-model", response=None, error=None):
        self.organization = provider
        self.model = model
        self.response = response or ChatResponse(content="ok", model=model)
        self.error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class SequenceFakeAdapter:
    def __init__(self, outcomes, *, provider="fake", model="fake-model"):
        self.organization = provider
        self.model = model
        self.outcomes = list(outcomes)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def make_llm(adapter, *, timeout_s=None, backup=None):
    routes = [Route("primary", adapter, RoutePolicy(timeout_s=timeout_s))]
    if backup is not None:
        routes.append(Route("backup", backup))
    return ResilientLLM(RecoveryPlan(routes))


def test_resilient_llm_delegates_to_first_route_and_returns_compatible_response():
    adapter = FakeAdapter()
    llm = make_llm(adapter)
    messages = [{"role": "user", "content": "hello"}]

    response = llm.chat(messages, temperature=0.2, tools=["tool"])

    assert isinstance(response, ResilientChatResponse)
    assert isinstance(response, ChatResponse)
    assert response.selected_route == "primary"
    assert response.attempts[0].success is True
    assert adapter.calls[0]["messages"] is messages
    assert adapter.calls[0]["temperature"] == 0.2
    assert adapter.calls[0]["tools"] == ["tool"]


def test_open_route_is_skipped_and_backup_route_handles_request():
    clock = FakeClock()
    primary_breaker = CircuitBreaker(
        failure_threshold=1,
        cooldown_s=30,
        clock=clock,
    )
    primary_breaker.record_failure()
    primary = FakeAdapter(provider="openai")
    backup = FakeAdapter(provider="anthropic")
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("primary", primary, breaker=primary_breaker),
                Route("backup", backup),
            ]
        )
    )

    response = llm.chat([])

    assert response.selected_route == "backup"
    assert primary.calls == []
    assert len(backup.calls) == 1
    assert [attempt.route_name for attempt in response.attempts] == ["backup"]


def test_failed_route_opens_breaker_and_is_skipped_on_next_request():
    primary_breaker = CircuitBreaker(failure_threshold=1, cooldown_s=30)
    primary = SequenceFakeAdapter(
        [LLMAPITimeoutError()],
        provider="openai",
        model="primary-model",
    )
    backup = FakeAdapter(provider="anthropic", model="backup-model")
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("primary", primary, breaker=primary_breaker),
                Route("backup", backup),
            ]
        )
    )

    first_response = llm.chat([])
    second_response = llm.chat([])

    assert first_response.selected_route == "backup"
    assert second_response.selected_route == "backup"
    assert len(primary.calls) == 1
    assert len(backup.calls) == 2
    assert primary_breaker.state is CircuitState.OPEN


def test_chat_exposes_circuit_events_and_last_events():
    clock = FakeClock()
    primary_breaker = CircuitBreaker(
        failure_threshold=1,
        cooldown_s=10,
        clock=clock,
    )
    primary = SequenceFakeAdapter(
        [
            LLMAPITimeoutError(),
            ChatResponse(content="primary recovered", model="primary-model"),
        ],
        provider="openai",
        model="primary-model",
    )
    backup = FakeAdapter(provider="anthropic", model="backup-model")
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("primary", primary, breaker=primary_breaker),
                Route("backup", backup),
            ]
        )
    )

    first_response = llm.chat([])
    second_response = llm.chat([])
    clock.advance(10)
    third_response = llm.chat([])

    assert [event.event_type for event in first_response.events] == ["opened"]
    assert [event.event_type for event in second_response.events] == ["skipped"]
    assert [event.event_type for event in third_response.events] == [
        "half_open",
        "closed",
    ]
    assert first_response.events[0].route_name == "primary"
    assert first_response.events[0].provider == "openai"
    assert first_response.events[0].model == "primary-model"
    assert first_response.events[0].error_type == "LLMAPITimeoutError"
    assert llm.last_events == third_response.events


def test_breaker_opening_stops_remaining_retries_for_current_route():
    primary_breaker = CircuitBreaker(failure_threshold=1, cooldown_s=30)
    primary = SequenceFakeAdapter(
        [LLMAPITimeoutError(), LLMAPITimeoutError()],
        provider="openai",
    )
    backup = FakeAdapter(provider="anthropic")
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route(
                    "primary",
                    primary,
                    RoutePolicy(max_attempts=3),
                    primary_breaker,
                ),
                Route("backup", backup),
            ]
        )
    )

    response = llm.chat([])

    assert response.selected_route == "backup"
    assert len(primary.calls) == 1
    assert [attempt.route_name for attempt in response.attempts] == [
        "primary",
        "backup",
    ]


def test_half_open_route_is_probed_and_closes_after_success():
    clock = FakeClock()
    primary_breaker = CircuitBreaker(
        failure_threshold=1,
        cooldown_s=10,
        clock=clock,
    )
    primary_breaker.record_failure()
    primary = FakeAdapter(provider="openai")
    backup = FakeAdapter(provider="anthropic")
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("primary", primary, breaker=primary_breaker),
                Route("backup", backup),
            ]
        )
    )

    clock.advance(10)
    response = llm.chat([])

    assert response.selected_route == "primary"
    assert len(primary.calls) == 1
    assert backup.calls == []
    assert primary_breaker.state is CircuitState.CLOSED


def test_all_open_routes_raise_safe_circuit_error_without_api_calls():
    clock = FakeClock()
    primary_breaker = CircuitBreaker(
        failure_threshold=1,
        cooldown_s=30,
        clock=clock,
    )
    backup_breaker = CircuitBreaker(
        failure_threshold=1,
        cooldown_s=60,
        clock=clock,
    )
    primary_breaker.record_failure()
    backup_breaker.record_failure()
    primary = FakeAdapter(provider="openai")
    backup = FakeAdapter(provider="anthropic")
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("primary", primary, breaker=primary_breaker),
                Route("backup", backup, breaker=backup_breaker),
            ]
        )
    )

    with pytest.raises(CircuitOpenError, match="circuit is open") as error:
        llm.chat([])

    assert error.value.cooldown_remaining_s == pytest.approx(30)
    assert primary.calls == []
    assert backup.calls == []
    assert llm.last_attempts == ()


def test_resilient_llm_forwards_chat_kwargs_without_mutating_original_kwargs():
    adapter = FakeAdapter()
    llm = make_llm(adapter)
    request_kwargs = {
        "tools": ["tool"],
        "tool_choice": "auto",
        "json_schema": {"type": "object"},
        "response_model": object,
        "previous_response": object(),
    }

    llm.chat([], **request_kwargs)

    assert adapter.calls[0]["tools"] is request_kwargs["tools"]
    assert adapter.calls[0]["tool_choice"] == "auto"
    assert adapter.calls[0]["json_schema"] is request_kwargs["json_schema"]
    assert adapter.calls[0]["response_model"] is object
    assert adapter.calls[0]["previous_response"] is request_kwargs["previous_response"]
    assert "timeout_s" not in request_kwargs


def test_route_timeout_is_added_only_when_user_did_not_provide_one():
    adapter = FakeAdapter()
    llm = make_llm(adapter, timeout_s=12.0)

    llm.chat([])
    assert adapter.calls[-1]["timeout_s"] == 12.0

    llm.chat([], timeout_s=3.0)
    assert adapter.calls[-1]["timeout_s"] == 3.0


def test_resilient_llm_reraises_original_adapter_error_and_records_failure():
    error = RuntimeError("adapter failed")
    adapter = FakeAdapter(error=error)
    llm = make_llm(adapter)

    with pytest.raises(RuntimeError) as raised:
        llm.chat([])

    assert raised.value is error
    assert len(llm.last_attempts) == 1
    assert llm.last_attempts[0].success is False
    assert llm.last_attempts[0].error_type == "RuntimeError"
    assert llm.last_attempts[0].error_message == "adapter failed"


def test_resilient_llm_does_not_fail_over_builtin_non_retryable_error():
    primary = FakeAdapter(error=TimeoutError("primary failed"))
    backup = FakeAdapter()
    llm = make_llm(primary, backup=backup)

    with pytest.raises(TimeoutError):
        llm.chat([])

    assert len(primary.calls) == 1
    assert backup.calls == []


def test_resilient_llm_retries_same_route_and_returns_all_attempts(monkeypatch):
    success = ChatResponse(content="ok", model="primary-model")
    primary = SequenceFakeAdapter(
        [LLMAPIRateLimitError(detail="busy"), success],
        provider="openai",
        model="primary-model",
    )
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route(
                    "primary",
                    primary,
                    RoutePolicy(max_attempts=2, backoff_s=0.25),
                )
            ]
        )
    )
    sleeps = []
    monkeypatch.setattr("llm_api_resilience.resilient_llm.sleep", sleeps.append)

    response = llm.chat([])

    assert response.selected_route == "primary"
    assert [attempt.success for attempt in response.attempts] == [False, True]
    assert [attempt.route_name for attempt in response.attempts] == [
        "primary",
        "primary",
    ]
    assert len(llm.last_attempts) == 2
    assert sleeps == [0.25]
    assert len(primary.calls) == 2


def test_resilient_llm_fails_over_after_route_attempts_are_exhausted():
    primary = SequenceFakeAdapter(
        [LLMAPITimeoutError(), LLMAPIServerError()],
        provider="openai",
        model="primary-model",
    )
    backup_response = ChatResponse(content="backup", model="backup-model")
    backup = SequenceFakeAdapter(
        [backup_response],
        provider="anthropic",
        model="backup-model",
    )
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("primary", primary, RoutePolicy(max_attempts=2)),
                Route("backup", backup),
            ]
        )
    )

    response = llm.chat([])

    assert response.content == "backup"
    assert response.selected_route == "backup"
    assert [attempt.route_name for attempt in response.attempts] == [
        "primary",
        "primary",
        "backup",
    ]
    assert response.attempts[-1].success is True
    assert len(primary.calls) == 2
    assert len(backup.calls) == 1


def test_resilient_llm_raises_aggregate_error_after_all_routes_fail():
    last_error = LLMAPIServerError(
        detail="api_key=secret request_body={'prompt': 'private'}"
    )
    primary = SequenceFakeAdapter(
        [LLMAPITimeoutError()],
        provider="openai",
        model="primary-model",
    )
    backup = SequenceFakeAdapter(
        [last_error],
        provider="anthropic",
        model="backup-model",
    )
    llm = ResilientLLM(
        RecoveryPlan([Route("primary", primary), Route("backup", backup)])
    )

    with pytest.raises(FailoverExhaustedError) as raised:
        llm.chat([])

    error = raised.value
    assert error.last_error is last_error
    assert error.last_exception is last_error
    assert error.__cause__ is last_error
    assert isinstance(error.attempts, tuple)
    assert len(error.attempts) == 2
    assert [attempt.route_name for attempt in error.attempts] == [
        "primary",
        "backup",
    ]
    assert "primary" in str(error)
    assert "openai/primary-model" in str(error)
    assert "anthropic/backup-model" in str(error)
    assert "LLMAPITimeoutError" in str(error)
    assert "LLMAPIServerError" in str(error)
    assert "secret" not in str(error)
    assert "request_body" not in str(error)


def test_resilient_llm_does_not_fail_over_non_retryable_errors():
    primary = SequenceFakeAdapter([LLMAPIAuthorizationError()])
    backup = SequenceFakeAdapter([ChatResponse(content="backup", model="backup")])
    llm = ResilientLLM(
        RecoveryPlan([Route("primary", primary), Route("backup", backup)])
    )

    with pytest.raises(LLMAPIAuthorizationError):
        llm.chat([])

    assert len(primary.calls) == 1
    assert backup.calls == []
    assert len(llm.last_attempts) == 1
    assert llm.last_attempts[0].success is False


def test_resilient_llm_uses_custom_classifier_for_retry_decisions():
    class RetryValueErrorClassifier:
        def is_retryable(self, error):
            return isinstance(error, ValueError)

    primary = SequenceFakeAdapter(
        [ValueError("temporary"), ChatResponse(content="ok", model="primary")]
    )
    llm = ResilientLLM(
        RecoveryPlan([Route("primary", primary, RoutePolicy(max_attempts=2))]),
        failure_classifier=RetryValueErrorClassifier(),
    )

    response = llm.chat([])

    assert response.content == "ok"
    assert len(primary.calls) == 2


def test_resilient_llm_does_not_sleep_after_last_attempt(monkeypatch):
    primary = SequenceFakeAdapter(
        [LLMAPITimeoutError(), LLMAPITimeoutError()],
    )
    backup = SequenceFakeAdapter([ChatResponse(content="backup", model="backup")])
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route(
                    "primary",
                    primary,
                    RoutePolicy(max_attempts=2, backoff_s=1.0),
                ),
                Route("backup", backup),
            ]
        )
    )
    sleeps = []
    monkeypatch.setattr("llm_api_resilience.resilient_llm.sleep", sleeps.append)

    llm.chat([])

    assert sleeps == [1.0]


def test_resilient_llm_keeps_previous_response_only_with_same_route():
    previous_response = ChatResponse(content="previous", model="primary")
    primary = SequenceFakeAdapter(
        [LLMAPITimeoutError(), LLMAPITimeoutError()],
        provider="openai",
        model="primary",
    )
    backup = SequenceFakeAdapter(
        [ChatResponse(content="backup", model="backup")],
        provider="anthropic",
        model="backup",
    )
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("primary", primary, RoutePolicy(max_attempts=2)),
                Route("backup", backup),
            ]
        )
    )

    llm.chat([], previous_response=previous_response)

    assert primary.calls[0]["previous_response"] is previous_response
    assert primary.calls[1]["previous_response"] is previous_response
    assert "previous_response" not in backup.calls[0]


@pytest.mark.parametrize(
    "first_error, second_error",
    [
        (LLMAPIRateLimitError(), LLMAPITimeoutError()),
        (LLMAPITimeoutError(), LLMAPIServerError()),
    ],
)
def test_failover_error_sequences_return_first_successful_response(
    first_error,
    second_error,
):
    primary = SequenceFakeAdapter(
        [first_error],
        provider="openai",
        model="primary-model",
    )
    secondary = SequenceFakeAdapter(
        [second_error],
        provider="anthropic",
        model="secondary-model",
    )
    final = SequenceFakeAdapter(
        [ChatResponse(content="success", model="final-model")],
        provider="google",
        model="final-model",
    )
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("primary", primary),
                Route("secondary", secondary),
                Route("final", final),
            ]
        )
    )

    response = llm.chat([{"role": "user", "content": "hello"}])

    assert isinstance(response, ResilientChatResponse)
    assert response.content == "success"
    assert response.selected_route == "final"
    assert [attempt.route_name for attempt in response.attempts] == [
        "primary",
        "secondary",
        "final",
    ]
    assert [attempt.error_type for attempt in response.attempts] == [
        type(first_error).__name__,
        type(second_error).__name__,
        None,
    ]
    assert [attempt.success for attempt in response.attempts] == [
        False,
        False,
        True,
    ]


def test_one_route_with_three_attempts_records_every_failure(monkeypatch):
    primary = SequenceFakeAdapter(
        [LLMAPITimeoutError(), LLMAPIRateLimitError(), LLMAPIServerError()],
        provider="openai",
        model="primary-model",
    )
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route(
                    "primary",
                    primary,
                    RoutePolicy(
                        max_attempts=3,
                        backoff_s=0.1,
                        backoff_multiplier=2.0,
                    ),
                )
            ]
        )
    )
    sleeps = []
    monkeypatch.setattr("llm_api_resilience.resilient_llm.sleep", sleeps.append)

    with pytest.raises(FailoverExhaustedError) as raised:
        llm.chat([])

    assert len(primary.calls) == 3
    assert len(raised.value.attempts) == 3
    assert [attempt.error_type for attempt in raised.value.attempts] == [
        "LLMAPITimeoutError",
        "LLMAPIRateLimitError",
        "LLMAPIServerError",
    ]
    assert sleeps == [0.1, 0.2]


@pytest.mark.parametrize(
    "error",
    [
        LLMAPIAuthorizationError(),
        LLMConfigError(),
        InvalidToolSchemaError(),
        JSONSchemaError(),
    ],
)
def test_non_retryable_errors_do_not_start_failover(error):
    primary = SequenceFakeAdapter(
        [error],
        provider="openai",
        model="primary-model",
    )
    backup = SequenceFakeAdapter(
        [ChatResponse(content="backup", model="backup-model")],
        provider="anthropic",
        model="backup-model",
    )
    llm = ResilientLLM(
        RecoveryPlan([Route("primary", primary), Route("backup", backup)])
    )

    with pytest.raises(type(error)):
        llm.chat([])

    assert len(primary.calls) == 1
    assert backup.calls == []


def test_tool_call_response_is_returned_without_automatic_replay():
    tool_response = ChatResponse(
        model="primary-model",
        tool_calls=[
            ToolCall(
                name="lookup_user",
                arguments={"user_id": "42"},
                call_id="call-1",
            )
        ],
        finish_reason="tool_calls",
    )
    primary = SequenceFakeAdapter(
        [tool_response],
        provider="openai",
        model="primary-model",
    )
    backup = SequenceFakeAdapter(
        [ChatResponse(content="backup", model="backup-model")],
        provider="anthropic",
        model="backup-model",
    )
    llm = ResilientLLM(
        RecoveryPlan([Route("primary", primary), Route("backup", backup)])
    )

    response = llm.chat(
        [{"role": "user", "content": "Find user 42"}],
        tools=["lookup_user"],
    )

    assert isinstance(response, ResilientChatResponse)
    assert response.tool_calls == tool_response.tool_calls
    assert response.finish_reason == "tool_calls"
    assert response.selected_route == "primary"
    assert len(primary.calls) == 1
    assert backup.calls == []

