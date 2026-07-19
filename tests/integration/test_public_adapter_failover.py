import pytest

from llm_api_adapter.llms.openai import sync_client as openai_sync_client
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

from llm_api_resilience import (
    RecoveryPlan,
    ResilientChatResponse,
    ResilientLLM,
    Route,
    RoutePolicy,
)

pytestmark = pytest.mark.integration


class FakeHTTPResponse:
    def __init__(self, *, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            error = openai_sync_client.requests.exceptions.HTTPError(
                f"HTTP {self.status_code}"
            )
            error.response = self
            raise error
        return None

    def json(self):
        if self.payload is not None:
            return self.payload
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


def test_universal_adapter_smoke_with_mocked_http(monkeypatch):
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


def test_universal_adapters_failover_with_mocked_http(monkeypatch):
    requests_seen = []
    mocked_responses = [
        FakeHTTPResponse(
            status_code=429,
            payload={
                "error": {
                    "type": "RateLimitError",
                    "message": "rate limited",
                }
            },
        ),
        FakeHTTPResponse(
            payload={
                "id": "response-backup",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "backup response",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            }
        ),
    ]

    def fake_post(url, headers, json, timeout):
        requests_seen.append(
            {
                "url": url,
                "headers": headers,
                "payload": json,
                "timeout": timeout,
            }
        )
        return mocked_responses.pop(0)

    monkeypatch.setattr(openai_sync_client.requests, "post", fake_post)

    primary = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-4o",
        api_key="test-primary-key",
    )
    backup = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-4o-mini",
        api_key="test-backup-key",
    )
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("primary", primary),
                Route("backup", backup),
            ]
        )
    )

    response = llm.chat(
        [{"role": "user", "content": "hello"}],
        timeout_s=2.5,
    )

    assert response.content == "backup response"
    assert response.selected_route == "backup"
    assert [attempt.error_type for attempt in response.attempts] == [
        "LLMAPIRateLimitError",
        None,
    ]
    assert [attempt.success for attempt in response.attempts] == [False, True]
    assert len(requests_seen) == 2
    assert requests_seen[0]["payload"]["model"] == "gpt-4o"
    assert requests_seen[1]["payload"]["model"] == "gpt-4o-mini"
    assert requests_seen[0]["payload"]["messages"] == requests_seen[1][
        "payload"
    ]["messages"]
    assert requests_seen[0]["timeout"] == 2.5
    assert requests_seen[1]["timeout"] == 2.5
