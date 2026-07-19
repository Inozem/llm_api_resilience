"""Route and recovery-plan contracts."""

from dataclasses import dataclass, field
from typing import Any, Iterator, Tuple

from .policies import RoutePolicy


@dataclass(frozen=True)
class Route:
    """A named adapter and the policy used to execute it."""

    name: str
    adapter: Any
    policy: RoutePolicy = field(default_factory=RoutePolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("route name must be a string")
        if not self.name.strip():
            raise ValueError("route name must not be empty")

        if not callable(getattr(self.adapter, "chat", None)):
            raise TypeError("adapter must provide a callable chat method")
        if not isinstance(self.policy, RoutePolicy):
            raise TypeError("policy must be a RoutePolicy")


@dataclass(frozen=True)
class RecoveryPlan:
    """An immutable, ordered collection of unique routes."""

    routes: Tuple[Route, ...]

    def __post_init__(self) -> None:
        try:
            routes = tuple(self.routes)
        except TypeError as exc:
            raise TypeError("routes must be an iterable of Route objects") from exc

        if not routes:
            raise ValueError("recovery plan must contain at least one route")
        if any(not isinstance(route, Route) for route in routes):
            raise TypeError("recovery plan routes must be Route objects")

        names = [route.name for route in routes]
        if len(names) != len(set(names)):
            raise ValueError("route names must be unique")

        object.__setattr__(self, "routes", routes)

    def __iter__(self) -> Iterator[Route]:
        return iter(self.routes)

    def __len__(self) -> int:
        return len(self.routes)

    def __getitem__(self, index: int) -> Route:
        return self.routes[index]
