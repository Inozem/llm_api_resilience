import pytest

from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

from llm_api_resilience import (
    RecoveryPlan,
    ResilientChatResponse,
    ResilientLLM,
    Route,
    RoutePolicy,
)


pytestmark = pytest.mark.e2e


def test_current_adapter_model_works_through_resilient_llm(configured_provider):
    provider = configured_provider["name"]
    model = configured_provider["model"]
    adapter = UniversalLLMAPIAdapter(
        organization=provider,
        model=model,
        api_key=configured_provider["api_key"],
    )
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route(
                    f"{provider}-latest",
                    adapter,
                    RoutePolicy(timeout_s=60),
                )
            ]
        )
    )

    response = llm.chat(
        [{"role": "user", "content": "Reply with exactly: E2E_OK"}],
        max_tokens=64,
    )

    assert isinstance(response, ResilientChatResponse)
    assert isinstance(response, ChatResponse)
    assert response.content and response.content.strip()
    assert response.selected_route == f"{provider}-latest"
    assert len(response.attempts) == 1
    assert response.attempts[0].provider == provider
    assert response.attempts[0].model == model
    assert response.attempts[0].success is True
