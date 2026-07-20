"""Safe observability records for resilience decisions."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .circuit_breaker import CircuitState


_CIRCUIT_EVENT_TYPES = frozenset(
    {"opened", "half_open", "closed", "skipped"}
)


@dataclass(frozen=True)
class CircuitEvent:
    """Safe metadata describing a circuit-breaker transition or skip."""

    event_type: str
    route_name: str
    state: CircuitState
    provider: Optional[str] = None
    model: Optional[str] = None
    error_type: Optional[str] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    cooldown_remaining_s: float = 0.0

    def __post_init__(self) -> None:
        if self.event_type not in _CIRCUIT_EVENT_TYPES:
            raise ValueError(
                "event_type must be one of: opened, half_open, closed, skipped"
            )
        if not isinstance(self.route_name, str):
            raise TypeError("route_name must be a string")
        if not self.route_name.strip():
            raise ValueError("route_name must not be empty")
        if not isinstance(self.state, CircuitState):
            raise TypeError("state must be a CircuitState")

        for field_name, value in (
            ("provider", self.provider),
            ("model", self.model),
            ("error_type", self.error_type),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} must not be empty")

        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        if isinstance(self.cooldown_remaining_s, bool) or not isinstance(
            self.cooldown_remaining_s, (int, float)
        ):
            raise TypeError("cooldown_remaining_s must be a non-negative number")
        if self.cooldown_remaining_s < 0:
            raise ValueError("cooldown_remaining_s must be non-negative")
