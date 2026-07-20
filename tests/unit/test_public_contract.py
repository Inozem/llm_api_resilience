import pytest

from llm_api_resilience import (
    AdapterProtocol,
    AttemptRecord,
    CapabilityMismatchError,
    CapabilityRequirements,
    CapabilitySkipEvent,
    CircuitBreaker,
    CircuitEvent,
    CircuitOpenError,
    CircuitSnapshot,
    CircuitState,
    DefaultFailureClassifier,
    FailoverExhaustedError,
    FailureClassifier,
    InvalidResultError,
    NoCompatibleRouteError,
    PromptProfile,
    RecoveryPlan,
    ResilientChatResponse,
    ResilientLLM,
    Route,
    RoutePolicy,
    RouteCapabilities,
    ResultDecision,
    ResultPolicy,
    ResultPolicyCallback,
    evaluate_result_policy,
)

pytestmark = pytest.mark.unit


def test_v01_public_exports_are_available():
    assert AdapterProtocol is not None
    assert AttemptRecord is not None
    assert CapabilityMismatchError is not None
    assert CapabilityRequirements is not None
    assert CapabilitySkipEvent is not None
    assert CircuitBreaker is not None
    assert CircuitEvent is not None
    assert CircuitOpenError is not None
    assert CircuitSnapshot is not None
    assert CircuitState is not None
    assert DefaultFailureClassifier is not None
    assert FailoverExhaustedError is not None
    assert FailureClassifier is not None
    assert InvalidResultError is not None
    assert NoCompatibleRouteError is not None
    assert PromptProfile is not None
    assert RecoveryPlan is not None
    assert ResilientChatResponse is not None
    assert ResilientLLM is not None
    assert Route is not None
    assert RoutePolicy is not None
    assert RouteCapabilities is not None
    assert ResultDecision is not None
    assert ResultPolicy is not None
    assert ResultPolicyCallback is not None
    assert evaluate_result_policy is not None
