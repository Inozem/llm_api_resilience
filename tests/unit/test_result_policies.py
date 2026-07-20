from dataclasses import FrozenInstanceError

import pytest

from llm_api_adapter.models.responses.chat_response import ChatResponse

from llm_api_resilience import (
    InvalidResultError,
    RecoveryPlan,
    ResultDecision,
    ResultPolicy,
    ResilientLLM,
    Route,
    evaluate_result_policy,
)


pytestmark = pytest.mark.unit


class FakeAdapter:
    organization = "openai"
    model = "gpt-test"

    def chat(self, **kwargs):
        return ChatResponse(content="ok", model=self.model)


class ObjectPolicy:
    def validate(self, response):
        return ResultDecision(valid=not response.content.startswith("bad"))


def test_result_decision_is_immutable_and_validates_its_fields():
    decision = ResultDecision(valid=False, reason_type="empty_content")

    assert decision.valid is False
    assert decision.reason_type == "empty_content"
    with pytest.raises(FrozenInstanceError):
        decision.valid = True


def test_result_policy_protocol_accepts_an_object_without_inheritance():
    policy = ObjectPolicy()

    assert isinstance(policy, ResultPolicy)
    assert evaluate_result_policy(
        policy,
        ChatResponse(content="ok"),
    ) == ResultDecision(valid=True)
    assert evaluate_result_policy(
        policy,
        ChatResponse(content="bad"),
    ) == ResultDecision(valid=False)


def test_callback_policy_is_supported_without_inheritance():
    def policy(response):
        return ResultDecision(
            valid=bool(response.content),
            reason_type="empty_content",
        )

    assert evaluate_result_policy(
        policy,
        ChatResponse(content="ok"),
    ).valid is True
    assert evaluate_result_policy(
        policy,
        ChatResponse(content=""),
    ).reason_type == "empty_content"


@pytest.mark.parametrize("value", [None, "invalid", 1, object()])
def test_policy_rejects_unsupported_decision_values(value):
    with pytest.raises(TypeError, match="boolean or ResultDecision"):
        evaluate_result_policy(
            lambda response: value,
            ChatResponse(content="ok"),
        )


def test_resilient_llm_accepts_an_optional_result_policy():
    policy = ObjectPolicy()
    llm = ResilientLLM(
        RecoveryPlan([Route("primary", FakeAdapter())]),
        result_policy=policy,
    )

    assert llm.result_policy is policy


def test_invalid_result_error_contains_only_safe_route_metadata():
    error = InvalidResultError(
        "primary",
        provider="openai",
        model="gpt-test",
        reason_type="schema_validation_failed",
    )

    assert error.route_name == "primary"
    assert error.provider == "openai"
    assert error.model == "gpt-test"
    assert error.reason_type == "schema_validation_failed"
    assert "openai/gpt-test" in str(error)
    assert "schema_validation_failed" in str(error)
    assert "response" not in vars(error)
    assert "request" not in str(error)


def test_invalid_result_error_validates_route_metadata():
    with pytest.raises(ValueError, match="route_name must not be empty"):
        InvalidResultError(" ")
    with pytest.raises(TypeError, match="reason_type must be a string"):
        InvalidResultError("primary", reason_type=object())


def test_resilient_llm_rejects_invalid_result_policy_configuration():
    with pytest.raises(TypeError, match="callable or provide"):
        ResilientLLM(
            RecoveryPlan([Route("primary", FakeAdapter())]),
            result_policy=object(),
        )
