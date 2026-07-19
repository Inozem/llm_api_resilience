import pytest

from llm_api_adapter.errors import (
    JSONSchemaError,
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from llm_api_adapter.errors.config_errors import LLMConfigError
from llm_api_adapter.errors.llm_api_error import (
    InvalidToolArgumentsError,
    InvalidToolSchemaError,
    ToolChoiceError,
)

from llm_api_resilience import (
    DefaultFailureClassifier,
    FailureClassifier,
    RecoveryPlan,
    ResilientLLM,
    Route,
)

pytestmark = pytest.mark.unit


class FakeAdapter:
    def chat(self, **kwargs):
        raise LLMAPITimeoutError()


@pytest.mark.parametrize(
    "error",
    [
        LLMAPITimeoutError(),
        LLMAPIRateLimitError(),
        LLMAPIServerError(),
    ],
)
def test_default_classifier_marks_transient_adapter_errors_as_retryable(error):
    classifier = DefaultFailureClassifier()

    assert classifier.is_retryable(error) is True


@pytest.mark.parametrize(
    "error",
    [
        LLMAPIAuthorizationError(),
        LLMConfigError(),
        InvalidToolSchemaError(),
        InvalidToolArgumentsError(),
        ToolChoiceError(),
        JSONSchemaError(),
        ValueError("invalid input"),
    ],
)
def test_default_classifier_does_not_retry_non_retryable_errors(error):
    classifier = DefaultFailureClassifier()

    assert classifier.is_retryable(error) is False


def test_failure_classifier_protocol_accepts_classifier_implementation():
    classifier = DefaultFailureClassifier()

    assert isinstance(classifier, FailureClassifier)


class AlwaysRetryClassifier:
    def is_retryable(self, error):
        return True


def test_resilient_llm_accepts_custom_failure_classifier():
    classifier = AlwaysRetryClassifier()
    llm = ResilientLLM(
        RecoveryPlan([Route("primary", FakeAdapter())]),
        failure_classifier=classifier,
    )

    assert llm.failure_classifier is classifier
    assert llm.failure_classifier.is_retryable(ValueError("custom")) is True


def test_resilient_llm_rejects_invalid_failure_classifier():
    with pytest.raises(TypeError, match="is_retryable method"):
        ResilientLLM(
            RecoveryPlan([Route("primary", FakeAdapter())]),
            failure_classifier=object(),
        )
