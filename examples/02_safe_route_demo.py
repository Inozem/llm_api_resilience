"""First offline SafeRoute demo built on top of ``llm-api-resilience``.

The example simulates a primary provider outage, then shows how the
resilience layer moves the request to a backup provider. It uses the same
``session`` API as the live examples and never makes a network request.

Run from the workspace root::

    python example_scripts/02_safe_route_demo.py
"""

from llm_api_resilience import CircuitBreaker, RecoveryPlan, ResilientLLM, Route

from fake_llm import build_safe_route_adapters


def main() -> None:
    messages = [
        {
            "role": "user",
            "content": "Prepare a short incident summary. Secret: [REDACTED].",
        }
    ]

    primary_breaker = CircuitBreaker(failure_threshold=1, cooldown_s=30)
    primary, backup = build_safe_route_adapters()

    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("primary", primary, breaker=primary_breaker),
                Route("backup", backup),
            ]
        )
    )

    session = llm.session(messages)
    response = session.start()

    print("SafeRoute offline demo")
    print(f"Selected route: {response.selected_route}")
    print(f"Answer: {response.content}")
    print("Attempts:")
    for attempt in session.attempts:
        status = "ok" if attempt.success else f"error={attempt.error_type}"
        print(f"  - {attempt.route_name}: {status}")

    print("Safe observability events:")
    for event in session.events:
        print(f"  - {event.route_name}: {event.event_type} -> {event.state.value}")

    print(f"Primary circuit state: {primary_breaker.state.value}")


if __name__ == "__main__":
    main()
