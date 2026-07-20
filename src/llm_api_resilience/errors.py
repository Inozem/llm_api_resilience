"""Errors raised by the resilience layer."""

from typing import Iterable, Optional, Tuple

from .attempts import AttemptRecord


class SessionStateError(RuntimeError):
    """Raised when a tool session operation is invalid for its current state."""


class CircuitOpenError(RuntimeError):
    """Raised when a circuit breaker rejects a request."""

    def __init__(self, cooldown_remaining_s: float = 0.0) -> None:
        if isinstance(cooldown_remaining_s, bool) or not isinstance(
            cooldown_remaining_s, (int, float)
        ):
            raise TypeError("cooldown_remaining_s must be a non-negative number")
        if cooldown_remaining_s < 0:
            raise ValueError("cooldown_remaining_s must be non-negative")

        self.cooldown_remaining_s = float(cooldown_remaining_s)
        super().__init__(
            "circuit is open; retry in "
            f"{self.cooldown_remaining_s:.3f} seconds"
        )


class InvalidResultError(RuntimeError):
    """Raised when an opted-in result policy rejects a model response."""

    def __init__(
        self,
        route_name: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        reason_type: str = "invalid_result",
    ) -> None:
        if not isinstance(route_name, str):
            raise TypeError("route_name must be a string")
        if not route_name.strip():
            raise ValueError("route_name must not be empty")
        for field_name, value in (("provider", provider), ("model", model)):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
        if not isinstance(reason_type, str):
            raise TypeError("reason_type must be a string")
        if not reason_type.strip():
            raise ValueError("reason_type must not be empty")

        self.route_name = route_name
        self.provider = provider
        self.model = model
        self.reason_type = reason_type
        provider_name = provider or "unknown-provider"
        model_name = model or "unknown-model"
        super().__init__(
            f"invalid result from {route_name} "
            f"[{provider_name}/{model_name}]: {reason_type}"
        )


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
