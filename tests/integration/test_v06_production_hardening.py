import pytest

from llm_api_adapter.llms.openai import sync_client as openai_sync_client
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

from llm_api_resilience import (
    CircuitBreaker,
    CircuitOpenError,
    FailoverExhaustedError,
    RecoveryPlan,
    ResilientLLM,
    Route,
    RoutePolicy,
    ToolResult,
)


pytestmark = pytest.mark.integration


class FakeHTTPResponse:
    def __init__(self, payload, *, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            error = openai_sync_client.requests.exceptions.HTTPError(
                f"HTTP {self.status_code}"
            )
            error.response = self
            raise error

    def json(self):
        return self.payload


def response_payload(content, *, model):
    return {
        "id": f"response-{model}",
        "model": model,
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        },
    }


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


def test_public_adapter_retry_uses_injected_sleeper(monkeypatch):
    requests_seen = []
    mocked_responses = [
        FakeHTTPResponse(
            {
                "error": {
                    "type": "RateLimitError",
                    "message": "rate limited",
                }
            },
            status_code=429,
        ),
        FakeHTTPResponse(response_payload("recovered", model="gpt-4o-mini")),
    ]

    def fake_post(url, headers, json, timeout):
        requests_seen.append(json)
        return mocked_responses.pop(0)

    monkeypatch.setattr(openai_sync_client.requests, "post", fake_post)
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-4o-mini",
        api_key="test-key",
    )
    sleeps = []
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route(
                    "primary",
                    adapter,
                    RoutePolicy(max_attempts=2, backoff_s=0.75),
                )
            ]
        ),
        sleeper=sleeps.append,
    )

    response = llm.chat([{"role": "user", "content": "hello"}])

    assert response.content == "recovered"
    assert response.selected_route == "primary"
    assert len(requests_seen) == 2
    assert sleeps == [0.75]


def test_public_adapter_tool_calls_work_through_resilient_session(monkeypatch):
    requests_seen = []
    mocked_responses = [
        FakeHTTPResponse(
            {
                "id": "tool-response",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Tel Aviv"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            }
        ),
        FakeHTTPResponse(response_payload("It is sunny.", model="gpt-4o-mini")),
    ]

    def fake_post(url, headers, json, timeout):
        requests_seen.append(json)
        return mocked_responses.pop(0)

    monkeypatch.setattr(openai_sync_client.requests, "post", fake_post)
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-4o-mini",
        api_key="test-key",
    )
    llm = ResilientLLM(RecoveryPlan([Route("primary", adapter)]))
    session = llm.session(
        [{"role": "user", "content": "What is the weather?"}],
        tools=[
            ToolSpec(
                name="get_weather",
                description="Get the weather for a city",
                json_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ],
        tool_choice="auto",
        max_tokens=128,
    )

    first = session.start()
    final = session.continue_with(ToolResult("call-1", "sunny"))

    assert isinstance(first, ChatResponse)
    assert first.tool_calls[0].name == "get_weather"
    assert first.tool_calls[0].arguments == {"city": "Tel Aviv"}
    assert final.content == "It is sunny."
    assert final.selected_route == "primary"
    assert requests_seen[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "sunny",
    }


def test_public_adapter_structured_output_metadata_is_preserved(monkeypatch):
    captured_request = {}

    def fake_post(url, headers, json, timeout):
        captured_request.update(json)
        return FakeHTTPResponse(
            response_payload('{"answer":"ok"}', model="gpt-4o-mini")
        )

    monkeypatch.setattr(openai_sync_client.requests, "post", fake_post)
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-4o-mini",
        api_key="test-key",
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    llm = ResilientLLM(RecoveryPlan([Route("primary", adapter)]))

    response = llm.chat(
        [{"role": "user", "content": "Return JSON."}],
        json_schema=schema,
        max_tokens=64,
    )

    assert response.parsed_json == {"answer": "ok"}
    assert response.selected_route == "primary"
    assert captured_request["response_format"]["type"] == "json_schema"


def test_public_adapter_breaker_skips_network_after_failure(monkeypatch):
    requests_seen = []

    def fake_post(url, headers, json, timeout):
        requests_seen.append(json)
        return FakeHTTPResponse(
            {
                "error": {
                    "type": "RateLimitError",
                    "message": "rate limited",
                }
            },
            status_code=429,
        )

    monkeypatch.setattr(openai_sync_client.requests, "post", fake_post)
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-4o-mini",
        api_key="test-key",
    )
    breaker = CircuitBreaker(
        failure_threshold=1,
        cooldown_s=30,
        clock=FakeClock(),
    )
    llm = ResilientLLM(
        RecoveryPlan([Route("primary", adapter, breaker=breaker)])
    )

    with pytest.raises(FailoverExhaustedError):
        llm.chat([{"role": "user", "content": "hello"}])

    with pytest.raises(CircuitOpenError):
        llm.chat([{"role": "user", "content": "hello again"}])

    assert len(requests_seen) == 1
