import pytest

from llm_api_adapter.models.responses.chat_response import ChatResponse

from llm_api_resilience import Checkpoint, RouteIdentity


pytestmark = pytest.mark.unit


def test_checkpoint_keeps_a_defensive_snapshot_of_messages_and_kwargs():
    messages = [{"role": "user", "content": "hello"}]
    tools = [
        {
            "name": "get_weather",
            "parameters": {"type": "object", "required": ["city"]},
        }
    ]
    request_kwargs = {"tools": tools, "max_tokens": 128}

    checkpoint = Checkpoint.capture(
        messages=messages,
        request_kwargs=request_kwargs,
        route=RouteIdentity("primary", "openai", "gpt-5"),
        operation_id="operation-1",
    )

    messages[0]["content"] = "changed"
    tools[0]["name"] = "changed"
    request_kwargs["max_tokens"] = 1

    assert checkpoint.messages == (
        {"role": "user", "content": "hello"},
    )
    assert checkpoint.request_kwargs == {
        "tools": [
            {
                "name": "get_weather",
                "parameters": {"type": "object", "required": ["city"]},
            }
        ],
        "max_tokens": 128,
    }

    returned_messages = checkpoint.messages
    returned_messages[0]["content"] = "mutated outside checkpoint"
    assert checkpoint.messages[0]["content"] == "hello"


def test_checkpoint_drops_provider_specific_previous_response():
    previous_response = ChatResponse(response_id="response-1")

    checkpoint = Checkpoint.capture(
        messages=[{"role": "user", "content": "hello"}],
        request_kwargs={
            "previous_response": previous_response,
            "max_tokens": 128,
        },
        route=RouteIdentity("primary", "openai", "gpt-5"),
    )

    assert checkpoint.request_kwargs == {"max_tokens": 128}


def test_route_identity_exposes_provider_model_compatibility():
    openai_primary = RouteIdentity("primary", "openai", "gpt-5")
    openai_backup = RouteIdentity("backup", "openai", "gpt-5")
    anthropic = RouteIdentity("anthropic", "anthropic", "claude-sonnet")

    assert openai_primary.provider_model == ("openai", "gpt-5")
    assert openai_primary.is_compatible_with(openai_backup)
    assert not openai_primary.is_compatible_with(anthropic)


def test_checkpoint_repr_does_not_include_request_values_or_secrets():
    checkpoint = Checkpoint.capture(
        messages=[{"role": "user", "content": "private request secret"}],
        request_kwargs={"system": "private parameter secret"},
        route=RouteIdentity("primary", "openai", "gpt-5"),
        operation_id="operation-1",
    )

    rendered = repr(checkpoint)

    assert "private request secret" not in rendered
    assert "private parameter secret" not in rendered
    assert "operation-1" in rendered


def test_checkpoint_rejects_credentials_in_request_kwargs_without_echoing_value():
    with pytest.raises(ValueError, match="must not contain credentials") as error:
        Checkpoint.capture(
            messages=[],
            request_kwargs={"api_key": "super-secret"},
            route=RouteIdentity("primary", "openai", "gpt-5"),
        )

    assert "super-secret" not in str(error.value)
