"""Demonstrate circuit-breaker failover and safe observability offline.

Run from the repository root::

    python examples/circuit_breaker_observability.py

The example uses scripted adapters, so it does not require API keys or network
access. The primary route fails once, is opened, gets skipped on the next
request, and then recovers through a half-open probe after the cooldown.
"""

from typing import Any, List

from llm_api_adapter.errors import LLMAPITimeoutError
from llm_api_adapter.models.responses.chat_response import ChatResponse

from llm_api_resilience import (
    CircuitBreaker,
    RecoveryPlan,
    ResilientChatResponse,
    ResilientLLM,
    Route,
)


class DemoClock:
    """Manual monotonic clock used to advance cooldown without sleeping."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class ScriptedAdapter:
    """Small adapter double that returns scripted responses or raises errors."""

    def __init__(self, organization: str, model: str, outcomes: List[Any]) -> None:
        self.organization = organization
        self.model = model
        self._outcomes = list(outcomes)

    def chat(self, **kwargs: Any) -> ChatResponse:
        if not self._outcomes:
            raise RuntimeError(f"No scripted outcome left for {self.organization}")

        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def print_response(label: str, response: ResilientChatResponse) -> None:
    print(f"\n{label}")
    print(f"Final route: {response.selected_route}")
    print(f"Answer: {response.content}")

    print("Attempts:")
    for attempt in response.attempts:
        status = "ok" if attempt.success else f"error={attempt.error_type}"
        print(f"  - {attempt.route_name}: {status}")

    print("Circuit events:")
    if not response.events:
        print("  - none")
    for event in response.events:
        details = f"{event.event_type} -> {event.state.value}"
        if event.error_type:
            details += f" ({event.error_type})"
        print(f"  - {event.route_name}: {details}")


def main() -> None:
    clock = DemoClock()
    primary_breaker = CircuitBreaker(
        failure_threshold=1,
        cooldown_s=10,
        clock=clock,
    )
    primary = ScriptedAdapter(
        "openai",
        "demo-primary",
        [
            LLMAPITimeoutError(detail="simulated provider outage"),
            ChatResponse(content="Primary recovered", model="demo-primary"),
        ],
    )
    backup = ScriptedAdapter(
        "anthropic",
        "demo-backup",
        [
            ChatResponse(content="Backup handled the request", model="demo-backup"),
            ChatResponse(content="Backup handled the request again", model="demo-backup"),
        ],
    )
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("primary", primary, breaker=primary_breaker),
                Route("backup", backup),
            ]
        )
    )

    first = llm.chat([{"role": "user", "content": "Request one"}])
    print_response("Request 1: primary fails and opens the circuit", first)

    second = llm.chat([{"role": "user", "content": "Request two"}])
    print_response("Request 2: open primary is skipped", second)

    clock.advance(10)
    third = llm.chat([{"role": "user", "content": "Request three"}])
    print_response("Request 3: cooldown expires and primary probe succeeds", third)
    print(f"\nPrimary circuit state: {primary_breaker.state.value}")


if __name__ == "__main__":
    main()
