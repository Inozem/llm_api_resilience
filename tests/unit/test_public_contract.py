from llm_api_adapter.llms.openai import sync_client as openai_sync_client
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

from llm_api_resilience import (
    AdapterProtocol,
    AttemptRecord,
    RecoveryPlan,
    ResilientChatResponse,
    ResilientLLM,
    Route,
    RoutePolicy,
)


def test_v01_public_exports_are_available():
    assert AdapterProtocol is not None
    assert AttemptRecord is not None
    assert RecoveryPlan is not None
    assert ResilientChatResponse is not None
    assert ResilientLLM is not None
    assert Route is not None
    assert RoutePolicy is not None


class FakeHTTPResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "id": "response-1",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello from mocked HTTP",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 4,
                "total_tokens": 6,
            },
        }


def test_integration_smoke_with_universal_adapter_and_mocked_http(monkeypatch):
    captured_request = {}

    def fake_post(url, headers, json, timeout):
        captured_request.update(
            url=url,
            headers=headers,
            payload=json,
            timeout=timeout,
        )
        return FakeHTTPResponse()

    monkeypatch.setattr(openai_sync_client.requests, "post", fake_post)

    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-4o-mini",
        api_key="test-key",
    )
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route(
                    "openai-primary",
                    adapter,
                    RoutePolicy(timeout_s=4.5),
                )
            ]
        )
    )

    response = llm.chat([{"role": "user", "content": "hello"}])

    assert isinstance(response, ResilientChatResponse)
    assert isinstance(response, ChatResponse)
    assert response.content == "hello from mocked HTTP"
    assert response.selected_route == "openai-primary"
    assert response.attempts[0].provider == "openai"
    assert response.attempts[0].model == "gpt-4o-mini"
    assert response.attempts[0].success is True
    assert captured_request["url"].endswith("/chat/completions")
    assert captured_request["timeout"] == 4.5
    assert captured_request["payload"]["messages"] == [
        {"role": "user", "content": "hello"}
    ]
