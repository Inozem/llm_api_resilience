"""Policies that control route execution."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RoutePolicy:
    """Execution settings associated with a route.

    Version 0.1 deliberately permits only one attempt.  Retry and backoff
    semantics are introduced by the v0.2 execution layer.
    """

    timeout_s: Optional[float] = None
    max_attempts: int = 1

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
        if self.max_attempts != 1:
            raise ValueError("v0.1 supports exactly one attempt per route")
