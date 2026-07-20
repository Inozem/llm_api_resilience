"""Run a real multi-provider tool loop with resilient failover.

Install the project with its example dependencies, configure the three
provider keys in the repository ``.env`` file, and run from the repository
root::

    python -m pip install -e ".[test]"
    python examples/multi_provider_tool_failover.py

The example demonstrates a provider-neutral tool-calling flow. The application
executes the tool, while ``ResilientSession`` handles the continuation and
replays the saved result if another route is needed.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from llm_api_adapter.models.messages.chat_message import Prompt, UserMessage
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

from llm_api_resilience import (
    RecoveryPlan,
    ResilientLLM,
    Route,
    RoutePolicy,
    ToolResult,
)


def _required_api_key(env_name: str) -> str:
    value = os.getenv(env_name)
    if not value or value.startswith("__REPLACE_WITH_"):
        raise RuntimeError(f"Set {env_name} in .env before running this example")
    return value


TOOLS = [
    ToolSpec(
        name="get_weather",
        description="Get current weather for a city",
        json_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    )
]


def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an application-owned tool.

    Replace this function with the real service call in your application.
    """

    if name == "get_weather":
        return {
            "city": args["city"],
            "temperature": 22,
            "unit": "C",
            "source": "demo-service",
        }
    raise ValueError(f"Unknown tool: {name}")


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    openai_adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5.6-sol",
        api_key=_required_api_key("OPENAI_API_KEY"),
    )
    anthropic_adapter = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-fable-5",
        api_key=_required_api_key("ANTHROPIC_API_KEY"),
    )
    google_adapter = UniversalLLMAPIAdapter(
        organization="google",
        model="gemini-3.5-flash",
        api_key=_required_api_key("GOOGLE_API_KEY"),
    )

    route_policy = RoutePolicy(max_attempts=1, timeout_s=60)
    llm = ResilientLLM(
        RecoveryPlan(
            [
                Route("openai", openai_adapter, route_policy),
                Route("anthropic", anthropic_adapter, route_policy),
                Route("google", google_adapter, route_policy),
            ]
        )
    )

    session = llm.session(
        [
            Prompt("If the user asks about weather, call get_weather."),
            UserMessage("What's the weather in Tel Aviv today?"),
        ],
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=1000,
        timeout_s=60,
    )

    response = session.start()
    print(f"Initial route: {response.selected_route}")
    tool_round = 0
    while response.tool_calls:
        tool_round += 1
        if tool_round > 8:
            raise RuntimeError("The model did not finish the tool loop")

        tool_results = []
        for tool_call in response.tool_calls:
            result = run_tool(tool_call.name, tool_call.arguments)
            tool_results.append(
                ToolResult(
                    tool_call_id=tool_call.call_id or tool_call.name,
                    content=json.dumps(result),
                )
            )
        response = session.continue_with(tool_results)

    print(f"Final route: {response.selected_route}")
    print(f"Answer: {response.content}")
    print("Attempts:")
    for attempt in session.attempts:
        status = "ok" if attempt.success else attempt.error_type
        print(f"  - {attempt.route_name}: {status}")
    print(f"Journal entries: {len(session.journal.entries)}")


if __name__ == "__main__":
    main()
