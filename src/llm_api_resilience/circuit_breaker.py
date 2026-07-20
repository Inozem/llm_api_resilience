"""Circuit-breaker state machine for route-level failure protection."""

from dataclasses import dataclass
from enum import Enum
import time
from typing import Callable, Optional

from .errors import CircuitOpenError


class CircuitState(str, Enum):
    """States used by a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class CircuitSnapshot:
    """Safe diagnostic metadata for the current breaker state."""

    state: CircuitState
    failure_count: int
    opened_at: Optional[float]
    cooldown_remaining_s: float


class CircuitBreaker:
    """Protect a route after a configured number of consecutive failures.

    The breaker starts in ``closed`` state. Once ``failure_threshold``
    failures are recorded, it enters ``open`` state and rejects requests for
    ``cooldown_s`` seconds. The first request after the cooldown is a single
    ``half_open`` probe. A successful probe closes the breaker; a failed
    probe opens it again and starts a new cooldown.

    ``clock`` is injectable so the state machine can be tested without
    sleeping. Thread-safety is intentionally outside this primitive and is
    planned for the production-hardening version.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_s: float = 30.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if isinstance(failure_threshold, bool) or not isinstance(
            failure_threshold, int
        ):
            raise TypeError("failure_threshold must be an integer")
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")

        if isinstance(cooldown_s, bool) or not isinstance(
            cooldown_s, (int, float)
        ):
            raise TypeError("cooldown_s must be a non-negative number")
        if cooldown_s < 0:
            raise ValueError("cooldown_s must be non-negative")

        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")

        self.failure_threshold = failure_threshold
        self.cooldown_s = float(cooldown_s)
        self._clock = clock or time.monotonic
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: Optional[float] = None
        self._probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        """Return the current state without reserving a probe request."""

        return self._state

    @property
    def failure_count(self) -> int:
        """Return the number of consecutive failures in closed state."""

        return self._failure_count

    def allow_request(self) -> bool:
        """Return whether a request may be sent through the circuit.

        An open circuit returns ``False`` until its cooldown expires. After
        that, exactly one caller receives ``True`` for the half-open probe;
        other callers are rejected until that probe is recorded as successful
        or failed.
        """

        if self._state is CircuitState.CLOSED:
            return True

        if self._state is CircuitState.OPEN:
            if self._cooldown_remaining() > 0:
                return False
            self._state = CircuitState.HALF_OPEN
            self._probe_in_flight = False

        if self._state is CircuitState.HALF_OPEN:
            if self._probe_in_flight:
                return False
            self._probe_in_flight = True
            return True

        return False

    def ensure_request_allowed(self) -> None:
        """Raise :class:`CircuitOpenError` when the circuit rejects a request."""

        if self.allow_request():
            return

        raise CircuitOpenError(
            cooldown_remaining_s=self._cooldown_remaining(),
        )

    def record_failure(self) -> None:
        """Record a failed request and update the breaker state."""

        if self._state is CircuitState.HALF_OPEN:
            self._failure_count = self.failure_threshold
            self._open()
            return

        if self._state is CircuitState.OPEN:
            return

        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._open()

    def record_success(self) -> None:
        """Record a successful request and close/reset the breaker."""

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = None
        self._probe_in_flight = False

    def reset(self) -> None:
        """Manually return the breaker to its initial closed state."""

        self.record_success()

    def snapshot(self) -> CircuitSnapshot:
        """Return safe state metadata without request or credential data."""

        return CircuitSnapshot(
            state=self._state,
            failure_count=self._failure_count,
            opened_at=self._opened_at,
            cooldown_remaining_s=self._cooldown_remaining(),
        )

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._now()
        self._probe_in_flight = False

    def _cooldown_remaining(self) -> float:
        if self._state is not CircuitState.OPEN or self._opened_at is None:
            return 0.0

        return max(0.0, self.cooldown_s - (self._now() - self._opened_at))

    def _now(self) -> float:
        return float(self._clock())
