"""Offline example for the v0.6 production-hardening contracts.

Run with::

    python examples/production_hardening.py

The example uses a fake adapter and an injected sleeper, so it does not make
network requests and does not require API keys.
"""

from llm_api_adapter.errors import LLMAPIRateLimitError
from llm_api_adapter.models.responses.chat_response import ChatResponse

from llm_api_resilience import (
    CircuitBreaker,
    RecoveryPlan,
    ResilientLLM,
    Route,
    RoutePolicy,
)


class SequenceAdapter:
    organization = "demo"
    model = "demo-model"

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)

    def chat(self, **kwargs):
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def main() -> None:
    sleeps = []
    breaker = CircuitBreaker(failure_threshold=3, cooldown_s=30)
    primary = SequenceAdapter(
        [
            LLMAPIRateLimitError(detail="temporary rate limit"),
            ChatResponse(content="Recovered on retry", model="demo-model"),
        ]
    )

    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route(
                    "primary",
                    primary,
                    policy=RoutePolicy(
                        max_attempts=2,
                        backoff_s=0.25,
                        backoff_multiplier=2.0,
                    ),
                    breaker=breaker,
                )
            ]
        ),
        sleeper=sleeps.append,
    )

    response = llm.chat([{"role": "user", "content": "Say hello."}])

    print(f"Selected route: {response.selected_route}")
    print(f"Answer: {response.content}")
    print(f"Attempts: {len(response.attempts)}")
    print(f"Injected sleeps: {sleeps}")
    print(f"Circuit state: {breaker.snapshot().state.value}")


if __name__ == "__main__":
    main()
