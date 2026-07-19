import pytest

from llm_api_adapter.errors import LLMAPITimeoutError
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools import ToolCall
from llm_api_adapter.models.messages.chat_message import AIMessage, ToolMessage

from llm_api_resilience import (
    RecoveryPlan,
    ResilientChatResponse,
    ResilientLLM,
    Route,
    SessionStateError,
    ToolResult,
)


pytestmark = pytest.mark.unit


class SequenceAdapter:
    organization = "openai"
    model = "gpt-test"

    def __init__(self, outcomes, *, provider=None, model=None):
        if provider is not None:
            self.organization = provider
        if model is not None:
            self.model = model
        self.outcomes = list(outcomes)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def tool_call_response(*, call_id="call-1", model="gpt-test"):
    return ChatResponse(
        model=model,
        tool_calls=[
            ToolCall(
                name="lookup_user",
                arguments={"user_id": "42"},
                call_id=call_id,
            )
        ],
        finish_reason="tool_calls",
    )


def make_llm(adapter):
    return ResilientLLM(RecoveryPlan([Route("primary", adapter)]))


def test_session_returns_tool_call_and_continues_with_application_result():
    adapter = SequenceAdapter(
        [tool_call_response(), ChatResponse(content="User is active", model="gpt-test")]
    )
    llm = make_llm(adapter)
    original_messages = [{"role": "user", "content": "Find user 42"}]
    session = llm.session(
        original_messages,
        tools=["lookup_user"],
        max_tokens=128,
    )

    first = session.start()
    final = session.continue_with(
        [ToolResult("call-1", '{"user_id":"42","active":true}')]
    )

    assert isinstance(first, ResilientChatResponse)
    assert first.tool_calls is not None
    assert final.content == "User is active"
    assert final.selected_route == "primary"
    assert session.is_closed is True
    assert session.checkpoint is not None
    assert session.checkpoint.route.provider_model == ("openai", "gpt-test")
    assert len(session.attempts) == 2
    assert original_messages == [{"role": "user", "content": "Find user 42"}]

    continuation_messages = adapter.calls[1]["messages"]
    assert isinstance(continuation_messages[1], AIMessage)
    assert isinstance(continuation_messages[2], ToolMessage)
    assert continuation_messages[2].tool_call_id == "call-1"
    assert adapter.calls[1]["previous_response"] is first


def test_session_can_accept_a_single_tool_result_object():
    adapter = SequenceAdapter(
        [tool_call_response(), ChatResponse(content="done", model="gpt-test")]
    )
    session = make_llm(adapter).session([])

    session.start()
    response = session.continue_with(ToolResult("call-1", "ok"))

    assert response.content == "done"


def test_session_without_tool_calls_is_closed_and_cannot_continue():
    adapter = SequenceAdapter([ChatResponse(content="final", model="gpt-test")])
    session = make_llm(adapter).session([])

    response = session.start()

    assert response.content == "final"
    assert session.checkpoint is None
    assert session.is_closed is True
    with pytest.raises(SessionStateError, match="final response"):
        session.continue_with([])


def test_session_requires_start_before_continuation():
    session = make_llm(SequenceAdapter([tool_call_response()])).session([])

    with pytest.raises(SessionStateError, match=r"start\(\)"):
        session.continue_with([])


@pytest.mark.parametrize(
    "tool_results",
    [
        [],
        [ToolResult("unknown", "ok")],
        [ToolResult("call-1", "first"), ToolResult("call-1", "duplicate")],
    ],
)
def test_session_rejects_tool_results_that_do_not_match_current_calls(tool_results):
    adapter = SequenceAdapter([tool_call_response()])
    session = make_llm(adapter).session([])
    session.start()

    with pytest.raises(ValueError):
        session.continue_with(tool_results)

    assert len(adapter.calls) == 1


def test_session_records_continuation_failure_and_reraises_original_error():
    error = LLMAPITimeoutError(detail="temporary")
    adapter = SequenceAdapter([tool_call_response(), error])
    llm = make_llm(adapter)
    session = llm.session([])
    session.start()

    with pytest.raises(LLMAPITimeoutError) as raised:
        session.continue_with(ToolResult("call-1", "ok"))

    assert raised.value is error
    assert [attempt.success for attempt in session.attempts] == [True, False]
    assert llm.last_attempts == session.attempts


def test_session_retries_same_continuation_without_duplicate_messages():
    error = LLMAPITimeoutError(detail="temporary")
    first = tool_call_response()
    final = ChatResponse(content="recovered", model="gpt-test")
    adapter = SequenceAdapter([first, error, final])
    llm = make_llm(adapter)
    session = llm.session([])
    first_response = session.start()
    result = ToolResult("call-1", "ok")

    with pytest.raises(LLMAPITimeoutError):
        session.continue_with(result)
    response = session.continue_with(result)

    assert response.content == "recovered"
    assert adapter.calls[1]["previous_response"] is first_response
    assert adapter.calls[2]["previous_response"] is first_response
    assert len(adapter.calls[1]["messages"]) == len(adapter.calls[2]["messages"])
    assert [attempt.success for attempt in session.attempts] == [True, False, True]


def test_session_updates_previous_response_for_each_same_route_tool_round():
    first = tool_call_response()
    second = ChatResponse(
        model="gpt-test",
        tool_calls=[
            ToolCall(
                name="lookup_user",
                arguments={"user_id": "42"},
                call_id="call-2",
            )
        ],
        finish_reason="tool_calls",
    )
    adapter = SequenceAdapter(
        [first, second, ChatResponse(content="done", model="gpt-test")]
    )
    session = make_llm(adapter).session([])

    first_response = session.start()
    second_response = session.continue_with(ToolResult("call-1", "first"))
    session.continue_with(ToolResult("call-2", "second"))

    assert adapter.calls[1]["previous_response"] is first_response
    assert adapter.calls[2]["previous_response"] is second_response


def test_session_replays_checkpoint_tool_result_on_next_route():
    timeout = LLMAPITimeoutError(detail="primary continuation failed")
    primary = SequenceAdapter([tool_call_response(), timeout])
    backup_first = tool_call_response(call_id="call-2", model="backup-test")
    backup = SequenceAdapter(
        [backup_first, ChatResponse(content="recovered", model="backup-test")],
        provider="anthropic",
        model="backup-test",
    )
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("primary", primary),
                Route("backup", backup),
            ]
        )
    )
    session = llm.session([{"role": "user", "content": "Find user 42"}])

    first = session.start()
    result = ToolResult(
        "call-1",
        "user is active",
        idempotency_key="lookup-user-42",
        replay_policy="side_effecting",
    )
    final = session.continue_with(result)

    assert final.content == "recovered"
    assert final.selected_route == "backup"
    assert [attempt.route_name for attempt in session.attempts] == [
        "primary",
        "primary",
        "backup",
        "backup",
    ]
    assert primary.calls[1]["previous_response"] is first
    assert "previous_response" not in backup.calls[0]
    assert backup.calls[1]["previous_response"] is not None
    assert backup.calls[1]["previous_response"].selected_route == "backup"
    assert len(session.journal.entries) == 1
    assert session.checkpoint.messages == (
        {"role": "user", "content": "Find user 42"},
    )
