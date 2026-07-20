import pytest

from llm_api_resilience import (
    AdapterProtocol,
    AttemptRecord,
    CircuitBreaker,
    CircuitOpenError,
    CircuitSnapshot,
    CircuitState,
    DefaultFailureClassifier,
    FailoverExhaustedError,
    FailureClassifier,
    RecoveryPlan,
    ResilientChatResponse,
    ResilientLLM,
    Route,
    RoutePolicy,
)

pytestmark = pytest.mark.unit


def test_v01_public_exports_are_available():
    assert AdapterProtocol is not None
    assert AttemptRecord is not None
    assert CircuitBreaker is not None
    assert CircuitOpenError is not None
    assert CircuitSnapshot is not None
    assert CircuitState is not None
    assert DefaultFailureClassifier is not None
    assert FailoverExhaustedError is not None
    assert FailureClassifier is not None
    assert RecoveryPlan is not None
    assert ResilientChatResponse is not None
    assert ResilientLLM is not None
    assert Route is not None
    assert RoutePolicy is not None
