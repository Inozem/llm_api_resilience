import pytest

from llm_api_adapter.llms.openai import sync_client as openai_sync_client
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

from llm_api_resilience import (
    PromptProfile,
    RecoveryPlan,
    ResilientLLM,
    Route,
    ResultDecision,
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


def test_public_adapter_receives_route_prompt_profile(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return FakeHTTPResponse(response_payload("ok", model="gpt-test"))

    monkeypatch.setattr(openai_sync_client.requests, "post", fake_post)
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-test",
        api_key="test-key",
    )
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route(
                    "primary",
                    adapter,
                    prompt_profile=PromptProfile(
                        system="Be concise.",
                        developer="Use plain language.",
                    ),
                )
            ]
        )
    )

    response = llm.chat([{"role": "user", "content": "Hello"}])

    assert response.content == "ok"
    assert captured["payload"]["messages"] == [
        {
            "role": "system",
            "content": "Be concise.\n\nDeveloper instructions:\nUse plain language.",
        },
        {"role": "user", "content": "Hello"},
    ]


def test_public_adapter_result_policy_failover_uses_backup(monkeypatch):
    requests_seen = []
    mocked_responses = [
        FakeHTTPResponse(response_payload("bad", model="gpt-primary")),
        FakeHTTPResponse(response_payload("valid", model="gpt-backup")),
    ]

    def fake_post(url, headers, json, timeout):
        requests_seen.append(json)
        return mocked_responses.pop(0)

    monkeypatch.setattr(openai_sync_client.requests, "post", fake_post)
    primary = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-primary",
        api_key="primary-key",
    )
    backup = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-backup",
        api_key="backup-key",
    )

    def policy(response):
        return ResultDecision(
            valid=response.content == "valid",
            reason_type="business_rule",
        )

    llm = ResilientLLM(
        RecoveryPlan([Route("primary", primary), Route("backup", backup)]),
        result_policy=policy,
        failover_on_invalid_result=True,
    )

    response = llm.chat([{"role": "user", "content": "Hello"}])

    assert response.selected_route == "backup"
    assert response.content == "valid"
    assert [request["model"] for request in requests_seen] == [
        "gpt-primary",
        "gpt-backup",
    ]
    assert response.attempts[0].error_type == "InvalidResultError"
