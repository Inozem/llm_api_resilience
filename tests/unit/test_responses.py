from llm_api_adapter.models.responses.chat_response import ChatResponse, Usage
from llm_api_adapter.models.tools import ToolCall

from llm_api_resilience import AttemptRecord, ResilientChatResponse


def make_base_response() -> ChatResponse:
    return ChatResponse(
        model="gpt-test",
        response_id="response-1",
        timestamp=123,
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        content="tool result pending",
        tool_calls=[
            ToolCall(
                name="lookup_user",
                arguments={"user_id": "42"},
                call_id="call-1",
            )
        ],
        parsed_json={"status": "ok"},
        parsed_model={"status": "ok"},
        cost_total=0.01,
        finish_reason="tool_calls",
    )


def test_resilient_response_is_chat_response_compatible():
    response = ResilientChatResponse.from_chat_response(
        make_base_response(),
        selected_route="primary",
    )

    assert isinstance(response, ResilientChatResponse)
    assert isinstance(response, ChatResponse)


def test_resilient_response_preserves_base_response_fields():
    base_response = make_base_response()
    response = ResilientChatResponse.from_chat_response(base_response)

    assert response.model == base_response.model
    assert response.response_id == base_response.response_id
    assert response.timestamp == base_response.timestamp
    assert response.usage == base_response.usage
    assert response.content == base_response.content
    assert response.cost_total == base_response.cost_total
    assert response.finish_reason == base_response.finish_reason


def test_resilient_response_contains_selected_route_and_attempts():
    attempt = AttemptRecord(
        route_name="primary",
        provider="openai",
        model="gpt-test",
        duration_s=0.4,
        success=True,
    )
    response = ResilientChatResponse.from_chat_response(
        make_base_response(),
        selected_route="primary",
        attempts=[attempt],
    )

    assert response.selected_route == "primary"
    assert response.attempts == (attempt,)


def test_resilient_response_does_not_drop_tool_calls_or_structured_output():
    base_response = make_base_response()
    response = ResilientChatResponse.from_chat_response(base_response)

    assert response.tool_calls == base_response.tool_calls
    assert response.parsed_json == base_response.parsed_json
    assert response.parsed_model == base_response.parsed_model
