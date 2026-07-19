"""Errors raised by the resilience layer."""

from typing import Iterable, Tuple

from .attempts import AttemptRecord


class FailoverExhaustedError(Exception):
    """Raised when every retryable route attempt has failed.

    The original final exception is retained for runtime inspection and as
    the exception cause.  The rendered message intentionally uses only safe
    attempt metadata and never includes request bodies or error messages.
    """

    def __init__(
        self,
        attempts: Iterable[AttemptRecord],
        last_error: Exception,
    ) -> None:
        normalized_attempts = tuple(attempts)
        if not normalized_attempts:
            raise ValueError("attempts must contain at least one record")
        if any(
            not isinstance(attempt, AttemptRecord)
            for attempt in normalized_attempts
        ):
            raise TypeError("attempts must contain AttemptRecord objects")
        if not isinstance(last_error, Exception):
            raise TypeError("last_error must be an Exception")

        self.attempts: Tuple[AttemptRecord, ...] = normalized_attempts
        self.last_error = last_error
        super().__init__(self._build_message())

    @property
    def last_exception(self) -> Exception:
        """Alias for the final exception retained by the aggregate error."""

        return self.last_error

    def _build_message(self) -> str:
        summaries = "; ".join(
            self._format_attempt(attempt) for attempt in self.attempts
        )
        return (
            f"Failover exhausted after {len(self.attempts)} attempts. "
            f"Last error: {type(self.last_error).__name__}. "
            f"Attempts: {summaries}"
        )

    @staticmethod
    def _format_attempt(attempt: AttemptRecord) -> str:
        provider = attempt.provider or "unknown-provider"
        model = attempt.model or "unknown-model"
        error_type = attempt.error_type or "UnknownError"
        return f"{attempt.route_name} [{provider}/{model}] -> {error_type}"
