"""Policies that control route execution."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RoutePolicy:
    """Execution settings associated with a route.

    The default values preserve the single-attempt behavior from v0.1.
    ``backoff_for`` receives the number of the failed attempt and returns the
    delay to apply before the next attempt.  It returns zero after the final
    configured attempt.
    """

    timeout_s: Optional[float] = None
    max_attempts: int = 1
    backoff_s: float = 0.0
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.timeout_s is not None:
            if isinstance(self.timeout_s, bool) or not isinstance(
                self.timeout_s, (int, float)
            ):
                raise TypeError("timeout_s must be a positive number or None")
            if self.timeout_s <= 0:
                raise ValueError("timeout_s must be greater than zero")

        if isinstance(self.max_attempts, bool) or not isinstance(
            self.max_attempts, int
        ):
            raise TypeError("max_attempts must be an integer")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        for field_name, value in (
            ("backoff_s", self.backoff_s),
            ("backoff_multiplier", self.backoff_multiplier),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a non-negative number")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")

    def backoff_for(self, failed_attempt: int) -> float:
        """Return the delay before the attempt after ``failed_attempt``.

        Attempt numbers are one-based.  No delay is returned when the failed
        attempt was the last allowed attempt.
        """

        if isinstance(failed_attempt, bool) or not isinstance(failed_attempt, int):
            raise TypeError("failed_attempt must be an integer")
        if failed_attempt < 1:
            raise ValueError("failed_attempt must be at least 1")
        if failed_attempt >= self.max_attempts:
            return 0.0

        return self.backoff_s * self.backoff_multiplier ** (failed_attempt - 1)
