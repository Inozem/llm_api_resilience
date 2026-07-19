"""Adapter contracts and execution-attempt metadata."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, runtime_checkable

from llm_api_adapter.models.responses.chat_response import ChatResponse


@runtime_checkable
class AdapterProtocol(Protocol):
    """Minimal adapter contract required by the resilience layer."""

    def chat(self, **kwargs: Any) -> ChatResponse:
        """Execute a chat request and return the adapter's normalized response."""
        ...


@dataclass(frozen=True)
class AttemptRecord:
    """Safe metadata for one route execution attempt.

    The record intentionally contains no request body, API key, raw response,
    or exception object.  Only the exception type and message are retained.
    """

    route_name: str
    provider: Optional[str] = None
    model: Optional[str] = None
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    duration_s: float = 0.0
    success: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.route_name, str):
            raise TypeError("route_name must be a string")
        if not self.route_name.strip():
            raise ValueError("route_name must not be empty")
        if not isinstance(self.started_at, datetime):
            raise TypeError("started_at must be a datetime")
        if isinstance(self.duration_s, bool) or not isinstance(
            self.duration_s, (int, float)
        ):
            raise TypeError("duration_s must be a non-negative number")
        if self.duration_s < 0:
            raise ValueError("duration_s must not be negative")
        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean")
