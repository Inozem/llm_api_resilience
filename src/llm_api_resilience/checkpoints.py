"""Provider-neutral snapshots for tool-calling recovery."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
from uuid import uuid4


_SENSITIVE_REQUEST_KEYS = frozenset(
    {
        "api_key",
        "anthropic_api_key",
        "authorization",
        "google_api_key",
        "headers",
        "openai_api_key",
    }
)


@dataclass(frozen=True)
class RouteIdentity:
    """Provider/model identity associated with a route snapshot."""

    name: str
    provider: Optional[str] = None
    model: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("route name must be a string")
        if not self.name.strip():
            raise ValueError("route name must not be empty")

        for field_name, value in (
            ("provider", self.provider),
            ("model", self.model),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} must not be empty")

    @property
    def provider_model(self) -> Tuple[Optional[str], Optional[str]]:
        """Return the identity used to decide provider-state compatibility."""

        return self.provider, self.model

    def is_compatible_with(self, other: "RouteIdentity") -> bool:
        """Return whether provider-specific response state can be reused."""

        if not isinstance(other, RouteIdentity):
            raise TypeError("other must be a RouteIdentity")
        return self.provider_model == other.provider_model


@dataclass(frozen=True, repr=False)
class Checkpoint:
    """Defensive snapshot of a request before its first tool round.

    The snapshot intentionally removes ``previous_response``. That value is
    provider-specific state and can only be reused by the session when the
    next attempt remains on a compatible route.
    """

    _messages: Tuple[Any, ...]
    _request_kwargs: Dict[str, Any]
    route: RouteIdentity
    operation_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not isinstance(self.route, RouteIdentity):
            raise TypeError("route must be a RouteIdentity")
        if not isinstance(self.operation_id, str):
            raise TypeError("operation_id must be a string")
        if not self.operation_id.strip():
            raise ValueError("operation_id must not be empty")

        try:
            messages = tuple(deepcopy(tuple(self._messages)))
        except TypeError as exc:
            raise TypeError("messages must be an iterable") from exc

        if not isinstance(self._request_kwargs, Mapping):
            raise TypeError("request_kwargs must be a mapping")
        request_kwargs = dict(self._request_kwargs)
        request_kwargs.pop("previous_response", None)
        self._validate_request_keys(request_kwargs)

        object.__setattr__(self, "_messages", messages)
        object.__setattr__(self, "_request_kwargs", deepcopy(request_kwargs))

    @classmethod
    def capture(
        cls,
        *,
        messages: Iterable[Any],
        request_kwargs: Mapping[str, Any],
        route: RouteIdentity,
        operation_id: Optional[str] = None,
    ) -> "Checkpoint":
        """Capture a request without retaining caller-owned mutable objects."""

        if isinstance(messages, (str, bytes)):
            raise TypeError("messages must be an iterable of message objects")
        return cls(
            _messages=tuple(messages),
            _request_kwargs=dict(request_kwargs),
            route=route,
            operation_id=operation_id or str(uuid4()),
        )

    @property
    def messages(self) -> Tuple[Any, ...]:
        """Return a fresh copy of the checkpoint messages."""

        return tuple(deepcopy(self._messages))

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        """Return a fresh copy of provider-neutral request parameters."""

        return deepcopy(self._request_kwargs)

    @staticmethod
    def _validate_request_keys(request_kwargs: Mapping[str, Any]) -> None:
        for key in request_kwargs:
            normalized_key = str(key).lower().replace("-", "_")
            if normalized_key in _SENSITIVE_REQUEST_KEYS:
                raise ValueError("checkpoint request_kwargs must not contain credentials")

    def __repr__(self) -> str:
        return (
            "Checkpoint("
            f"operation_id={self.operation_id!r}, "
            f"route={self.route!r}, "
            f"message_count={len(self._messages)}, "
            f"request_keys={tuple(sorted(self._request_kwargs))!r}"
            ")"
        )
