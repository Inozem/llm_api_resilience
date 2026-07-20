import pytest

from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools import ToolCall

from llm_api_resilience import (
    CircuitBreaker,
    FailoverExhaustedError,
    InvalidResultError,
    RecoveryPlan,
    ResultDecision,
    ResilientLLM,
    Route,
    RoutePolicy,
    ToolResult,
)


pytestmark = pytest.mark.unit


class SequenceAdapter:
    def __init__(self, outcomes, *, provider, model):
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


def result_policy(response):
    if response.content == "bad":
        return ResultDecision(valid=False, reason_type="business_rule")
    return True


def tool_call_response():
    return ChatResponse(
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


def make_llm(primary, backup, *, policy=RoutePolicy()):
    return ResilientLLM(
        RecoveryPlan(
            [
                Route("primary", primary, policy),
                Route("backup", backup),
            ]
        ),
        result_policy=result_policy,
        failover_on_invalid_result=True,
    )


def test_invalid_primary_result_fails_over_to_valid_backup():
    primary = SequenceAdapter(
        [ChatResponse(content="bad", model="primary-model")],
        provider="openai",
        model="primary-model",
    )
    backup = SequenceAdapter(
        [ChatResponse(content="valid", model="backup-model")],
        provider="anthropic",
        model="backup-model",
    )

    response = make_llm(primary, backup).chat([])

    assert response.content == "valid"
    assert response.selected_route == "backup"
    assert [attempt.success for attempt in response.attempts] == [False, True]
    assert response.attempts[0].error_type == "InvalidResultError"
    assert response.attempts[0].error_message == (
        "invalid result from primary [openai/primary-model]: business_rule"
    )
    assert len(primary.calls) == 1
    assert len(backup.calls) == 1


def test_invalid_result_can_retry_the_same_route_before_failover():
    primary = SequenceAdapter(
        [
            ChatResponse(content="bad", model="primary-model"),
            ChatResponse(content="valid", model="primary-model"),
        ],
        provider="openai",
        model="primary-model",
    )
    backup = SequenceAdapter(
        [ChatResponse(content="unused", model="backup-model")],
        provider="anthropic",
        model="backup-model",
    )

    response = make_llm(
        primary,
        backup,
        policy=RoutePolicy(max_attempts=2),
    ).chat([])

    assert response.selected_route == "primary"
    assert response.content == "valid"
    assert [attempt.success for attempt in response.attempts] == [False, True]
    assert len(backup.calls) == 0


def test_valid_result_does_not_trigger_backup():
    primary = SequenceAdapter(
        [ChatResponse(content="valid", model="primary-model")],
        provider="openai",
        model="primary-model",
    )
    backup = SequenceAdapter(
        [ChatResponse(content="unused", model="backup-model")],
        provider="anthropic",
        model="backup-model",
    )

    response = make_llm(primary, backup).chat([])

    assert response.selected_route == "primary"
    assert response.content == "valid"
    assert backup.calls == []


def test_structured_output_policy_rejects_missing_required_field():
    def structured_policy(response):
        payload = response.parsed_json
        is_valid = isinstance(payload, dict) and "answer" in payload
        return ResultDecision(
            valid=is_valid,
            reason_type="structured_output_missing_answer",
        )

    primary = SequenceAdapter(
        [
            ChatResponse(
                content='{"detail":"not enough"}',
                parsed_json={"detail": "not enough"},
                model="primary-model",
            )
        ],
        provider="openai",
        model="primary-model",
    )
    backup = SequenceAdapter(
        [
            ChatResponse(
                content='{"answer":"ready"}',
                parsed_json={"answer": "ready"},
                model="backup-model",
            )
        ],
        provider="anthropic",
        model="backup-model",
    )
    llm = ResilientLLM(
        RecoveryPlan([Route("primary", primary), Route("backup", backup)]),
        result_policy=structured_policy,
        failover_on_invalid_result=True,
    )

    response = llm.chat([])

    assert response.selected_route == "backup"
    assert response.parsed_json == {"answer": "ready"}
    assert response.attempts[0].error_message.endswith(
        "structured_output_missing_answer"
    )


def test_invalid_result_without_opt_in_is_returned_as_an_error():
    primary = SequenceAdapter(
        [ChatResponse(content="bad", model="primary-model")],
        provider="openai",
        model="primary-model",
    )
    backup = SequenceAdapter(
        [ChatResponse(content="valid", model="backup-model")],
        provider="anthropic",
        model="backup-model",
    )
    llm = ResilientLLM(
        RecoveryPlan([Route("primary", primary), Route("backup", backup)]),
        result_policy=result_policy,
    )

    with pytest.raises(InvalidResultError, match="business_rule"):
        llm.chat([])

    assert backup.calls == []
    assert len(llm.last_attempts) == 1
    assert llm.last_attempts[0].success is False


def test_invalid_result_does_not_open_the_route_breaker():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=30)
    primary = SequenceAdapter(
        [ChatResponse(content="bad", model="primary-model")],
        provider="openai",
        model="primary-model",
    )
    backup = SequenceAdapter(
        [ChatResponse(content="valid", model="backup-model")],
        provider="anthropic",
        model="backup-model",
    )
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("primary", primary, breaker=breaker),
                Route("backup", backup),
            ]
        ),
        result_policy=result_policy,
        failover_on_invalid_result=True,
    )

    llm.chat([])

    assert breaker.state.value == "closed"


def test_invalid_continuation_replays_checkpoint_on_backup_without_repeating_tools():
    primary = SequenceAdapter(
        [
            tool_call_response(),
            ChatResponse(content="bad", model="primary-model"),
        ],
        provider="openai",
        model="primary-model",
    )
    backup = SequenceAdapter(
        [ChatResponse(content="recovered", model="backup-model")],
        provider="anthropic",
        model="backup-model",
    )
    llm = make_llm(primary, backup)
    session = llm.session([{"role": "user", "content": "Find user 42"}])

    session.start()
    response = session.continue_with(ToolResult("call-1", "{\"active\": true}"))

    assert response.content == "recovered"
    assert response.selected_route == "backup"
    assert [attempt.success for attempt in response.attempts] == [True, False, True]
    assert backup.calls[0]["messages"] == [
        {"role": "user", "content": "Find user 42"},
    ]
    assert session.journal.entries[0].result.content == '{"active": true}'


def test_failover_flag_requires_a_result_policy():
    adapter = SequenceAdapter(
        [ChatResponse(content="ok")],
        provider="openai",
        model="primary-model",
    )

    with pytest.raises(ValueError, match="requires a result_policy"):
        ResilientLLM(
            RecoveryPlan([Route("primary", adapter)]),
            failover_on_invalid_result=True,
        )
