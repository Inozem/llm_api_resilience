import pytest

from llm_api_adapter.errors import (
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from llm_api_adapter.models.responses.chat_response import ChatResponse

from llm_api_resilience import (
    RecoveryPlan,
    ResilientChatResponse,
    ResilientLLM,
    Route,
    RoutePolicy,
)


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

