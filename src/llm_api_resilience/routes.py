"""Route and recovery-plan contracts."""

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Tuple

from .circuit_breaker import CircuitBreaker
from .policies import RoutePolicy
from .prompt_profiles import PromptProfile


@dataclass(frozen=True)
class Route:
    """A named adapter, execution policy, and optional prompt profile."""

    name: str
    adapter: Any
    policy: RoutePolicy = field(default_factory=RoutePolicy)
    breaker: Optional[CircuitBreaker] = None
    prompt_profile: Optional[PromptProfile] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("route name must be a string")
        if not self.name.strip():
            raise ValueError("route name must not be empty")

        if not callable(getattr(self.adapter, "chat", None)):
            raise TypeError("adapter must provide a callable chat method")
        if not isinstance(self.policy, RoutePolicy):
            raise TypeError("policy must be a RoutePolicy")
        if self.breaker is not None and not isinstance(
            self.breaker, CircuitBreaker
        ):
            raise TypeError("breaker must be a CircuitBreaker or None")
        if self.prompt_profile is not None and not isinstance(
            self.prompt_profile, PromptProfile
        ):
            raise TypeError("prompt_profile must be a PromptProfile or None")


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

    def reset_breakers(self) -> None:
        """Reset every configured route breaker in the recovery plan."""

        for route in self.routes:
            if route.breaker is not None:
                route.breaker.reset()
