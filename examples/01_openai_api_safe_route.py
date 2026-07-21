"""Live OpenAI SafeRoute demo.

GPT is the primary route. The offline adapter is kept as a deterministic
backup for retryable provider failures, so the demo still has a safe response
when the live provider is temporarily unavailable.

The API key is read from the repository ``.env`` file and is never printed.
This script makes a real API request; run ``00_judge_offline_demo.py`` for the
fully offline failover demonstration.

Run from the workspace root::

    python example_scripts/01_openai_api_safe_route.py
"""

from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

from llm_api_resilience import CircuitBreaker, RecoveryPlan, ResilientLLM, Route, RoutePolicy
from api_config import load_workspace_env, required_api_key
from fake_llm import build_safe_route_backup


def main() -> None:
    load_workspace_env()

    model = "gpt-5.6-sol"
    openai_adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model=model,
        api_key=required_api_key("OPENAI_API_KEY"),
    )

    primary_breaker = CircuitBreaker(failure_threshold=1, cooldown_s=30)
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route(
                    "gpt-primary",
                    openai_adapter,
                    policy=RoutePolicy(max_attempts=1, timeout_s=30),
                    breaker=primary_breaker,
                ),
                Route("offline-backup", build_safe_route_backup()),
            ]
        )
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise security assistant. Do not repeat secrets "
                "or personal data from the request."
            ),
        },
        {
            "role": "user",
            "content": (
                "Explain in two sentences why a resilient route is useful for "
                "a production LLM application."
            ),
        },
    ]

    session = llm.session(messages, max_tokens=200, temperature=0.2, timeout_s=30)
    response = session.start()

    print("SafeRoute live API demo")
    print(f"Configured GPT model: {model}")
    print(f"Selected route: {response.selected_route}")
    print(f"Answer: {response.content}")
    print("Attempts:")
    for attempt in session.attempts:
        status = "ok" if attempt.success else f"error={attempt.error_type}"
        print(f"  - {attempt.route_name}: {status}")

    print("Safe observability events:")
    for event in session.events:
        print(f"  - {event.route_name}: {event.event_type} -> {event.state.value}")

    print(f"GPT circuit state: {primary_breaker.state.value}")


if __name__ == "__main__":
    main()
