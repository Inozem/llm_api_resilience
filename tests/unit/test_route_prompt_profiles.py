import pytest

from llm_api_adapter.errors import LLMAPITimeoutError
from llm_api_adapter.models.messages.chat_message import Messages
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools import ToolCall

from llm_api_resilience import (
    RecoveryPlan,
    ResilientLLM,
    Route,
    PromptProfile,
    ToolResult,
)


pytestmark = pytest.mark.unit


class CaptureAdapter:
    organization = "fake"
    model = "fake-model"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def tool_call_response():
    return ChatResponse(
        model="fake-model",
        tool_calls=[
            ToolCall(
                name="lookup_user",
                arguments={"user_id": "42"},
                call_id="call-1",
            )
        ],
        finish_reason="tool_calls",
    )


def test_route_accepts_and_exposes_prompt_profile():
    profile = PromptProfile(system="Be concise.")
    route = Route("primary", CaptureAdapter([ChatResponse(content="ok")]), prompt_profile=profile)

    assert route.prompt_profile is profile


def test_route_rejects_invalid_prompt_profile():
    with pytest.raises(TypeError, match="PromptProfile or None"):
        Route("primary", CaptureAdapter([ChatResponse(content="ok")]), prompt_profile=object())


def test_profile_request_messages_are_accepted_by_llm_api_adapter():
    messages = PromptProfile(
        system="Be concise.",
        developer="Use plain language.",
    ).apply_to_request([{"role": "user", "content": "Hello"}])

    normalized = Messages(list(messages))

    assert normalized.to_openai() == [
        {
            "role": "system",
            "content": "Be concise.\n\nDeveloper instructions:\nUse plain language.",
        },
        {"role": "user", "content": "Hello"},
    ]


def test_profile_is_merged_with_an_existing_system_message():
    messages = PromptProfile(system="Route instructions").apply_to_request(
        [
            {"role": "system", "content": "Application instructions"},
            {"role": "user", "content": "Hello"},
        ]
    )

    assert messages == (
        {
            "role": "system",
            "content": "Route instructions\n\nApplication instructions",
        },
        {"role": "user", "content": "Hello"},
    )


def test_chat_applies_the_selected_route_profile_without_mutating_original_messages():
    primary = CaptureAdapter([LLMAPITimeoutError()])
    backup = CaptureAdapter([ChatResponse(content="backup")])
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route(
                    "primary",
                    primary,
                    prompt_profile=PromptProfile(system="Primary instructions"),
                ),
                Route(
                    "backup",
                    backup,
                    prompt_profile=PromptProfile(developer="Backup instructions"),
                ),
            ]
        )
    )
    messages = [{"role": "user", "content": "Hello"}]

    response = llm.chat(messages)

    assert response.selected_route == "backup"
    assert primary.calls[0]["messages"] == [
        {"role": "system", "content": "Primary instructions"},
        {"role": "user", "content": "Hello"},
    ]
    assert backup.calls[0]["messages"] == [
        {
            "role": "system",
            "content": "Developer instructions:\nBackup instructions",
        },
        {"role": "user", "content": "Hello"},
    ]
    assert messages == [{"role": "user", "content": "Hello"}]


def test_session_applies_profile_to_each_round_without_duplicating_it():
    adapter = CaptureAdapter(
        [tool_call_response(), ChatResponse(content="done", model="fake-model")]
    )
    profile = PromptProfile(system="Route instructions")
    llm = ResilientLLM(
        RecoveryPlan([Route("primary", adapter, prompt_profile=profile)])
    )
    session = llm.session([{"role": "user", "content": "Find user 42"}])

    session.start()
    session.continue_with(ToolResult("call-1", "{\"active\": true}"))

    first_messages = adapter.calls[0]["messages"]
    continuation_messages = adapter.calls[1]["messages"]
    assert first_messages.count({"role": "system", "content": "Route instructions"}) == 1
    assert continuation_messages.count(
        {"role": "system", "content": "Route instructions"}
    ) == 1
    assert session.checkpoint.messages == (
        {"role": "user", "content": "Find user 42"},
    )


def test_checkpoint_replay_uses_the_target_route_profile():
    primary = CaptureAdapter([tool_call_response(), LLMAPITimeoutError()])
    backup = CaptureAdapter([ChatResponse(content="recovered", model="fake-model")])
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route(
                    "primary",
                    primary,
                    prompt_profile=PromptProfile(system="Primary route"),
                ),
                Route(
                    "backup",
                    backup,
                    prompt_profile=PromptProfile(system="Backup route"),
                ),
            ]
        )
    )
    session = llm.session([{"role": "user", "content": "Find user 42"}])

    session.start()
    response = session.continue_with(ToolResult("call-1", "{\"active\": true}"))

    assert response.selected_route == "backup"
    assert backup.calls[0]["messages"][0] == {
        "role": "system",
        "content": "Backup route",
    }
    assert backup.calls[0]["messages"].count(
        {"role": "system", "content": "Backup route"}
    ) == 1
    assert session.checkpoint.messages == (
        {"role": "user", "content": "Find user 42"},
    )
