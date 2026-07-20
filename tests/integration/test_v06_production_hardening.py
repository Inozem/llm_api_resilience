import pytest

from llm_api_adapter.llms.openai import sync_client as openai_sync_client
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

from llm_api_resilience import (
    CircuitBreaker,
    CircuitOpenError,
    FailoverExhaustedError,
    RecoveryPlan,
    ResilientLLM,
    Route,
    RoutePolicy,
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
