"""Route-independent prompt profiles.

The profile deliberately deals in plain message dictionaries.  Provider
adapters can translate these messages at the request boundary without making
the resilience layer depend on a particular SDK representation.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True, repr=False)
class PromptProfile:
    """Immutable system and developer instructions for a route.

    Profile messages are returned before the caller's messages.  Every call
    creates fresh dictionaries and ``apply_to`` deep-copies the input, so
    adding a profile never mutates a request that may be reused for failover.
    """

    system: Optional[str] = None
    developer: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in ("system", "developer"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} must not be blank")

    def __repr__(self) -> str:
        """Avoid exposing instruction text in logs and tracebacks."""

        return (
            "PromptProfile("
            f"has_system={self.system is not None}, "
            f"has_developer={self.developer is not None}"
            ")"
        )

    @property
    def is_empty(self) -> bool:
        """Whether this profile adds no instructions."""

        return self.system is None and self.developer is None

    def to_messages(self) -> Tuple[Dict[str, str], ...]:
        """Return fresh provider-neutral messages for this profile."""

        messages = []
        if self.system is not None:
            messages.append({"role": "system", "content": self.system})
        if self.developer is not None:
            messages.append({"role": "developer", "content": self.developer})
        return tuple(messages)

    def to_request_messages(self) -> Tuple[Dict[str, str], ...]:
        """Return messages compatible with the current adapter contract.

        ``llm-api-adapter`` exposes one provider-neutral instruction role:
        ``system``.  Developer instructions are therefore kept explicit in
        the profile contract but represented as a labeled section inside one
        system message at the request boundary.
        """

        if self.is_empty:
            return ()

        sections = []
        if self.system is not None:
            sections.append(self.system)
        if self.developer is not None:
            sections.append(f"Developer instructions:\n{self.developer}")
        return ({"role": "system", "content": "\n\n".join(sections)},)

    def apply_to(
        self,
        messages: Iterable[Mapping[str, Any]],
    ) -> Tuple[Mapping[str, Any], ...]:
        """Prepend profile messages without mutating the input messages."""

        if isinstance(messages, (str, bytes, Mapping)):
            raise TypeError("messages must be an iterable of message mappings")
        return self.to_messages() + tuple(deepcopy(tuple(messages)))

    def apply_to_request(self, messages: Iterable[Any]) -> Tuple[Any, ...]:
        """Apply the profile using the adapter-compatible instruction role."""

        if isinstance(messages, (str, bytes, Mapping)):
            raise TypeError("messages must be an iterable of message objects")

        normalized = list(deepcopy(tuple(messages)))
        profile_messages = self.to_request_messages()
        if not profile_messages:
            return tuple(normalized)

        profile_message = profile_messages[0]
        for index, message in enumerate(normalized):
            role = (
                message.get("role")
                if isinstance(message, dict)
                else getattr(message, "role", None)
            )
            if role != "system":
                continue

            content = (
                message.get("content")
                if isinstance(message, dict)
                else getattr(message, "content", "")
            )
            normalized[index] = {
                "role": "system",
                "content": f"{profile_message['content']}\n\n{content}",
            }
            return tuple(normalized)

        return (profile_message, *normalized)
