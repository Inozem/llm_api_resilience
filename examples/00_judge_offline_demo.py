"""Deterministic SafeRoute demo for hackathon judges.

This is the recommended demo entry point. It uses fake provider connections,
but the real ``llm-api-resilience`` library and the same ``session`` API as the
live examples. It requires no API keys, network access, or external services.

Run from the workspace root::

    python example_scripts/00_judge_offline_demo.py
"""

from llm_api_resilience import CircuitBreaker, RecoveryPlan, ResilientLLM, Route

from fake_llm import build_safe_route_adapters


def main() -> None:
    fake_primary, fake_backup = build_safe_route_adapters()
    primary_breaker = CircuitBreaker(failure_threshold=1, cooldown_s=30)

    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("simulated-primary", fake_primary, breaker=primary_breaker),
                Route("simulated-backup", fake_backup),
            ]
        )
    )

    session = llm.session(
        [{"role": "user", "content": "Prepare a short safe incident summary."}]
    )
    response = session.start()

    assert response.selected_route == "simulated-backup"
    assert [attempt.route_name for attempt in session.attempts] == [
        "simulated-primary",
        "simulated-backup",
    ]

    print("SafeRoute judge demo")
    print("Mode: offline / fake connections / no API keys")
    print("Flow: simulated-primary -> simulated-backup")
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
    print("Demo verification: PASS")


if __name__ == "__main__":
    main()
