"""Response types returned by the resilience layer."""

from dataclasses import dataclass, fields, field
from typing import Iterable, Optional, Tuple

from llm_api_adapter.models.responses.chat_response import ChatResponse

from .attempts import AttemptRecord
from .observability import CircuitEvent


@dataclass
class ResilientChatResponse(ChatResponse):
    """A ``ChatResponse`` carrying route, attempt, and event metadata."""

    selected_route: Optional[str] = None
    attempts: Tuple[AttemptRecord, ...] = field(default_factory=tuple)
    events: Tuple[CircuitEvent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.selected_route is not None:
            if not isinstance(self.selected_route, str):
                raise TypeError("selected_route must be a string or None")
            if not self.selected_route.strip():
                raise ValueError("selected_route must not be empty")

        attempts = tuple(self.attempts)
        if any(not isinstance(attempt, AttemptRecord) for attempt in attempts):
            raise TypeError("attempts must contain AttemptRecord objects")
        self.attempts = attempts

        events = tuple(self.events)
        if any(not isinstance(event, CircuitEvent) for event in events):
            raise TypeError("events must contain CircuitEvent objects")
        self.events = events

    @classmethod
    def from_chat_response(
        cls,
        response: ChatResponse,
        *,
        selected_route: Optional[str] = None,
        attempts: Iterable[AttemptRecord] = (),
        events: Iterable[CircuitEvent] = (),
    ) -> "ResilientChatResponse":
        """Wrap an adapter response without dropping normalized response fields."""

        if not isinstance(response, ChatResponse):
            raise TypeError("response must be a ChatResponse")

        response_fields = {
            response_field.name: getattr(response, response_field.name)
            for response_field in fields(ChatResponse)
        }
        return cls(
            **response_fields,
            selected_route=selected_route,
            attempts=tuple(attempts),
            events=tuple(events),
        )
