import pytest

from llm_api_adapter.models.tools import ToolCall

from llm_api_resilience import (
    ReplayPolicy,
    ToolExecutionJournal,
    ToolResult,
)


pytestmark = pytest.mark.unit


def make_call(*, call_id="call-1", city="Tel Aviv"):
    return ToolCall(
        name="get_weather",
        arguments={"city": city},
        call_id=call_id,
    )


def test_journal_records_and_replays_a_read_only_tool_result():
    journal = ToolExecutionJournal()
    call = make_call()
    result = ToolResult(call.call_id, '{"temperature":22}')

    entry = journal.record(call, result)
    replayed = journal.replay_result(make_call(call_id="new-call"))

    assert entry.tool_call_id == "call-1"
    assert entry.tool_name == "get_weather"
    assert entry.arguments == {"city": "Tel Aviv"}
    assert entry.status == "completed"
    assert entry.replay_policy is ReplayPolicy.REPLAYABLE
    assert replayed == ToolResult("new-call", '{"temperature":22}')
    assert len(journal.entries) == 1


def test_journal_keeps_defensive_argument_copies():
    journal = ToolExecutionJournal()
    arguments = {"city": "Tel Aviv", "options": {"units": "C"}}
    call = ToolCall(name="get_weather", arguments=arguments, call_id="call-1")

    entry = journal.record(call, ToolResult("call-1", "ok"))
    arguments["options"]["units"] = "F"
    returned_arguments = entry.arguments
    returned_arguments["city"] = "London"

    assert entry.arguments == {
        "city": "Tel Aviv",
        "options": {"units": "C"},
    }


def test_journal_requires_idempotency_key_for_side_effecting_tools():
    journal = ToolExecutionJournal()

    with pytest.raises(ValueError, match="idempotency_key"):
        journal.record(
            ToolCall(
                name="send_email",
                arguments={"to": "user@example.com"},
                call_id="call-1",
            ),
            ToolResult("call-1", "sent"),
            replay_policy=ReplayPolicy.SIDE_EFFECTING,
        )


def test_journal_reuses_side_effect_result_by_idempotency_key():
    journal = ToolExecutionJournal()
    first_call = ToolCall(
        name="charge_card",
        arguments={"amount": 100},
        call_id="call-1",
    )
    second_call = ToolCall(
        name="charge_card",
        arguments={"amount": 100},
        call_id="call-2",
    )

    entry = journal.record(
        first_call,
        ToolResult("call-1", "charged", idempotency_key="payment-1"),
        replay_policy=ReplayPolicy.SIDE_EFFECTING,
    )
    replayed = journal.replay_result(second_call, idempotency_key="payment-1")

    assert journal.lookup(second_call, idempotency_key="payment-1") is entry
    assert replayed == ToolResult(
        "call-2",
        "charged",
        idempotency_key="payment-1",
        replay_policy=ReplayPolicy.SIDE_EFFECTING,
    )
    assert len(journal.entries) == 1


def test_journal_rejects_conflicting_idempotency_key():
    journal = ToolExecutionJournal()
    journal.record(
        ToolCall(name="charge_card", arguments={"amount": 100}, call_id="call-1"),
        ToolResult("call-1", "charged", idempotency_key="payment-1"),
        replay_policy=ReplayPolicy.SIDE_EFFECTING,
    )

    with pytest.raises(ValueError, match="another tool invocation"):
        journal.lookup(
            ToolCall(name="charge_card", arguments={"amount": 200}, call_id="call-2"),
            idempotency_key="payment-1",
        )


def test_journal_does_not_replay_failed_execution():
    journal = ToolExecutionJournal()
    call = make_call()

    journal.record(call, ToolResult("call-1", "failed"), status="failed")

    assert journal.lookup(call) is None
    assert journal.replay_result(call) is None


def test_journal_repr_does_not_include_arguments_or_result_values():
    journal = ToolExecutionJournal()
    journal.record(
        ToolCall(
            name="private_tool",
            arguments={"token": "secret-token"},
            call_id="call-1",
        ),
        ToolResult("call-1", "private-result"),
    )

    rendered = repr(journal) + repr(journal.entries[0])

    assert "secret-token" not in rendered
    assert "private-result" not in rendered
