"""Demonstrate safe tool-session failover without API keys.

The primary fake provider requests a side-effecting tool, then fails while
continuing the conversation. The backup provider requests the same tool, but
``ToolExecutionJournal`` replays the saved result instead of executing the
side effect a second time.

This is the strongest SafeRoute feature and is deterministic enough for a
hackathon video or a judge's machine.

Run from the workspace root::

    python example_scripts/03_tool_session_failover_demo.py
"""

import json
from typing import Any, Dict, List

from llm_api_adapter.models.tools import ToolSpec

from llm_api_resilience import (
    CircuitBreaker,
    RecoveryPlan,
    ResilientLLM,
    Route,
    ToolResult,
)

from fake_llm import build_tool_failover_adapters


TOOLS = [
    ToolSpec(
        name="create_incident_ticket",
        description="Create an incident ticket exactly once.",
        json_schema={
            "type": "object",
            "properties": {
                "incident_id": {"type": "string"},
                "severity": {"type": "string"},
            },
            "required": ["incident_id", "severity"],
            "additionalProperties": False,
        },
    )
]


def run_tool(name: str, arguments: Dict[str, Any], executions: List[str]) -> Dict[str, Any]:
    """Execute the application-owned side effect and record each execution."""

    if name != "create_incident_ticket":
        raise ValueError(f"Unknown tool: {name}")

    ticket_id = f"TICKET-{arguments['incident_id']}"
    executions.append(ticket_id)
    return {
        "ticket_id": ticket_id,
        "incident_id": arguments["incident_id"],
        "status": "created",
    }


def main() -> None:
    primary, backup = build_tool_failover_adapters()
    primary_breaker = CircuitBreaker(failure_threshold=1, cooldown_s=30)
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("primary", primary, breaker=primary_breaker),
                Route("backup", backup),
            ]
        )
    )

    executions: List[str] = []
    session = llm.session(
        [{"role": "user", "content": "Create a high-severity ticket for INC-42."}],
        tools=TOOLS,
        tool_choice="auto",
    )

    response = session.start()
    initial_route = response.selected_route
    tool_rounds = 0

    while response.tool_calls:
        tool_rounds += 1
        if tool_rounds > 3:
            raise RuntimeError("The demo tool loop did not finish")

        results = []
        for tool_call in response.tool_calls:
            tool_result = run_tool(tool_call.name, tool_call.arguments, executions)
            results.append(
                ToolResult(
                    tool_call_id=tool_call.call_id or tool_call.name,
                    content=json.dumps(tool_result),
                    idempotency_key=f"incident-ticket:{tool_call.arguments['incident_id']}",
                    replay_policy="side_effecting",
                )
            )
        response = session.continue_with(results)

    assert initial_route == "primary"
    assert response.selected_route == "backup"
    assert executions == ["TICKET-INC-42"]
    assert len(session.journal.entries) == 1
    assert session.checkpoint is not None

    print("SafeRoute tool-session failover demo")
    print("Mode: offline / fake connections / no API keys")
    print("Scenario: tool executes, continuation fails, backup replays the result")
    print(f"Initial route: {initial_route}")
    print(f"Final route: {response.selected_route}")
    print(f"Answer: {response.content}")
    print(f"Tool executions: {len(executions)} ({executions[0]})")
    print(f"Journal entries: {len(session.journal.entries)}")
    print("Attempts:")
    for attempt in session.attempts:
        status = "ok" if attempt.success else f"error={attempt.error_type}"
        print(f"  - {attempt.route_name}: {status}")

    print("Safe observability events:")
    for event in session.events:
        print(f"  - {event.route_name}: {event.event_type} -> {event.state.value}")

    print(f"Primary circuit state: {primary_breaker.state.value}")
    print("Verification: PASS - side effect executed exactly once")


if __name__ == "__main__":
    main()
