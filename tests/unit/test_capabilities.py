import pytest

from llm_api_adapter.models.responses.chat_response import ChatResponse

from llm_api_resilience import (
    CapabilityRequirements,
    CapabilitySkipEvent,
    NoCompatibleRouteError,
    RecoveryPlan,
    ResilientLLM,
    Route,
    RouteCapabilities,
    capability_names,
)


pytestmark = pytest.mark.unit


class CaptureAdapter:
    def __init__(self, *, provider, model):
        self.organization = provider
        self.model = model
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return ChatResponse(content="ok", model=self.model)


def test_capability_contract_exposes_stable_names_and_validates_flags():
    requirements = CapabilityRequirements(vision=True, structured_output=True)
    capabilities = RouteCapabilities(vision=True)

    assert capability_names() == ("reasoning", "vision", "structured_output")
    assert requirements.requested() == ("vision", "structured_output")
    assert capabilities.missing(requirements) == ("structured_output",)
    assert capabilities.supports(CapabilityRequirements(vision=True)) is True


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CapabilityRequirements(vision=1),
        lambda: RouteCapabilities(reasoning="yes"),
    ],
)
def test_capability_contract_rejects_non_boolean_flags(factory):
    with pytest.raises(TypeError, match="must be a boolean"):
        factory()


def test_incompatible_route_is_skipped_before_adapter_call():
    primary = CaptureAdapter(provider="openai", model="text-model")
    backup = CaptureAdapter(provider="google", model="vision-model")
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route(
                    "primary",
                    primary,
                    capabilities=RouteCapabilities(vision=False),
                ),
                Route(
                    "backup",
                    backup,
                    capabilities=RouteCapabilities(vision=True),
                ),
            ]
        )
    )

    response = llm.chat(
        [],
        capability_requirements=CapabilityRequirements(vision=True),
    )

    assert response.selected_route == "backup"
    assert primary.calls == []
    assert len(backup.calls) == 1
    assert isinstance(response.events[0], CapabilitySkipEvent)
    assert response.events[0].missing_capabilities == ("vision",)


def test_missing_capabilities_raise_safe_diagnostic_error():
    primary = CaptureAdapter(provider="openai", model="text-model")
    backup = CaptureAdapter(provider="anthropic", model="small-model")
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route(
                    "primary",
                    primary,
                    capabilities=RouteCapabilities(vision=False),
                ),
                Route(
                    "backup",
                    backup,
                    capabilities=RouteCapabilities(reasoning=True),
                ),
            ]
        )
    )

    with pytest.raises(NoCompatibleRouteError) as raised:
        llm.chat(
            [{"role": "user", "content": "private request"}],
            capability_requirements=CapabilityRequirements(vision=True),
        )

    error = raised.value
    assert error.requirements.vision is True
    assert [event.route_name for event in error.skipped_routes] == [
        "primary",
        "backup",
    ]
    assert primary.calls == []
    assert backup.calls == []
    assert "private request" not in str(error)
    assert "openai/text-model" in str(error)


def test_route_without_capability_metadata_keeps_backward_compatible_behavior():
    adapter = CaptureAdapter(provider="openai", model="unknown-capabilities")
    llm = ResilientLLM(RecoveryPlan([Route("primary", adapter)]))

    response = llm.chat(
        [],
        capability_requirements=CapabilityRequirements(
            reasoning=True,
            vision=True,
            structured_output=True,
        ),
    )

    assert response.selected_route == "primary"
    assert len(adapter.calls) == 1
    assert response.events == ()


def test_session_applies_capability_requirements_before_start_call():
    primary = CaptureAdapter(provider="openai", model="text-model")
    backup = CaptureAdapter(provider="google", model="vision-model")
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route(
                    "primary",
                    primary,
                    capabilities=RouteCapabilities(vision=False),
                ),
                Route(
                    "backup",
                    backup,
                    capabilities=RouteCapabilities(vision=True),
                ),
            ]
        )
    )

    response = llm.session(
        [],
        capability_requirements=CapabilityRequirements(vision=True),
    ).start()

    assert response.selected_route == "backup"
    assert primary.calls == []
    assert len(backup.calls) == 1
