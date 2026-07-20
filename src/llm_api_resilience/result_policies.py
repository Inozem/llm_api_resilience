"""Contracts for validating normalized LLM results."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Union, runtime_checkable

from llm_api_adapter.models.responses.chat_response import ChatResponse


@dataclass(frozen=True)
class ResultDecision:
    """The provider-neutral outcome of a result policy evaluation."""

    valid: bool
    reason_type: str = "invalid_result"

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a boolean")
        if not isinstance(self.reason_type, str):
            raise TypeError("reason_type must be a string")
        if not self.reason_type.strip():
            raise ValueError("reason_type must not be empty")


ResultPolicyResult = Union[bool, ResultDecision]


@runtime_checkable
class ResultPolicy(Protocol):
    """Structural contract for an object that validates a ChatResponse."""

    def validate(self, response: ChatResponse) -> ResultPolicyResult:
        """Return ``True``/``False`` or a detailed ``ResultDecision``."""
        ...


ResultPolicyCallback = Callable[[ChatResponse], ResultPolicyResult]


def normalize_result_policy(policy: Optional[Any]) -> Optional[Any]:
    """Validate and return a policy object or callback unchanged."""

    if policy is None:
        return None
    if callable(policy) or callable(getattr(policy, "validate", None)):
        return policy
    raise TypeError(
        "result_policy must be callable or provide a callable validate method"
    )


def evaluate_result_policy(
    policy: Any,
    response: ChatResponse,
) -> ResultDecision:
    """Evaluate a policy and normalize its return value."""

    if not isinstance(response, ChatResponse):
        raise TypeError("response must be a ChatResponse")

    normalized_policy = normalize_result_policy(policy)
    if normalized_policy is None:
        raise TypeError("result_policy must not be None")

    validator = getattr(normalized_policy, "validate", None)
    raw_decision = (
        validator(response)
        if callable(validator)
        else normalized_policy(response)
    )
    if isinstance(raw_decision, ResultDecision):
        return raw_decision
    if isinstance(raw_decision, bool):
        return ResultDecision(valid=raw_decision)
    raise TypeError(
        "result policy must return a boolean or ResultDecision"
    )
