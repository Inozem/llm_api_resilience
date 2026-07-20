"""Declarative route capability requirements."""

from dataclasses import dataclass
from typing import Optional, Tuple


_CAPABILITY_NAMES = ("reasoning", "vision", "structured_output")


def _validate_capability_flags(values) -> None:
    for name, value in values:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")


@dataclass(frozen=True)
class CapabilityRequirements:
    """Capabilities required by one chat or session request."""

    reasoning: bool = False
    vision: bool = False
    structured_output: bool = False

    def __post_init__(self) -> None:
        _validate_capability_flags(self._items())

    def requested(self) -> Tuple[str, ...]:
        """Return the requested capability names in stable order."""

        return tuple(name for name, enabled in self._items() if enabled)

    @property
    def is_empty(self) -> bool:
        return not bool(self.requested())

    def _items(self):
        return (
            ("reasoning", self.reasoning),
            ("vision", self.vision),
            ("structured_output", self.structured_output),
        )


@dataclass(frozen=True)
class RouteCapabilities:
    """Capabilities declared by an application for one route."""

    reasoning: bool = False
    vision: bool = False
    structured_output: bool = False

    def __post_init__(self) -> None:
        _validate_capability_flags(self._items())

    def missing(self, requirements: CapabilityRequirements) -> Tuple[str, ...]:
        """Return required capabilities not declared by this route."""

        if not isinstance(requirements, CapabilityRequirements):
            raise TypeError("requirements must be CapabilityRequirements")
        return tuple(
            name
            for name, required in requirements._items()
            if required and not getattr(self, name)
        )

    def supports(self, requirements: CapabilityRequirements) -> bool:
        return not self.missing(requirements)

    def _items(self):
        return (
            ("reasoning", self.reasoning),
            ("vision", self.vision),
            ("structured_output", self.structured_output),
        )


def normalize_capability_requirements(
    requirements: Optional[CapabilityRequirements],
) -> CapabilityRequirements:
    """Normalize omitted requirements while validating explicit values."""

    if requirements is None:
        return CapabilityRequirements()
    if not isinstance(requirements, CapabilityRequirements):
        raise TypeError(
            "capability_requirements must be CapabilityRequirements or None"
        )
    return requirements


def capability_names() -> Tuple[str, ...]:
    """Return the supported capability names."""

    return _CAPABILITY_NAMES
