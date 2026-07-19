"""Failure classification contracts for retry and failover decisions."""

from typing import Protocol, runtime_checkable

from llm_api_adapter.errors import (
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)


@runtime_checkable
class FailureClassifier(Protocol):
    """Classify an exception as retryable or non-retryable."""

    def is_retryable(self, error: Exception) -> bool:
        """Return whether the failed operation may be attempted again."""
        ...


class DefaultFailureClassifier:
    """Retry transient failures reported by the base LLM adapter."""

    retryable_errors = (
        LLMAPITimeoutError,
        LLMAPIRateLimitError,
        LLMAPIServerError,
    )

    def is_retryable(self, error: Exception) -> bool:
        """Return ``True`` only for timeout, rate-limit, and server errors."""

        return isinstance(error, self.retryable_errors)

