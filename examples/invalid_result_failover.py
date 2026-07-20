"""Demonstrate result-level failover without network access.

Run from the repository root::

    python examples/invalid_result_failover.py

Both adapters return valid structured JSON.  The primary result has low
confidence, so the result policy rejects it and the resilience layer selects
the higher-confidence backup response.
"""

from typing import Any, List

from llm_api_adapter.models.responses.chat_response import ChatResponse

from llm_api_resilience import (
    RecoveryPlan,
    ResultDecision,
    ResilientChatResponse,
    ResilientLLM,
    Route,
)


class ScriptedAdapter:
    """Small adapter double that returns scripted normalized responses."""

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


def validate_structured_result(response: ChatResponse) -> ResultDecision:
    """Accept only results that meet the application's quality threshold."""

    payload = response.parsed_json
    confidence = payload.get("confidence") if isinstance(payload, dict) else None
    valid = (
        isinstance(payload, dict)
        and isinstance(payload.get("answer"), str)
        and bool(payload["answer"].strip())
        and isinstance(confidence, (int, float))
        and confidence >= 0.8
        and payload.get("source") == "verified-demo-service"
    )
    return ResultDecision(
        valid=valid,
        reason_type="semantic_quality_threshold",
    )


def print_result(response: ResilientChatResponse) -> None:
    print(f"Final route: {response.selected_route}")
    print(f"Answer: {response.content}")
    print("Attempts:")
    for attempt in response.attempts:
        if attempt.success:
            print(f"  - {attempt.route_name}: ok")
        else:
            print(f"  - {attempt.route_name}: {attempt.error_message}")


def main() -> None:
    primary = ScriptedAdapter(
        "openai",
        "demo-primary",
        [
            ChatResponse(
                content='{"answer":"The service is probably available","confidence":0.42,"source":"unverified"}',
                parsed_json={
                    "answer": "The service is probably available",
                    "confidence": 0.42,
                    "source": "unverified",
                },
                model="demo-primary",
            )
        ],
    )
    backup = ScriptedAdapter(
        "anthropic",
        "demo-backup",
        [
            ChatResponse(
                content='{"answer":"The service is available","confidence":0.96,"source":"verified-demo-service"}',
                parsed_json={
                    "answer": "The service is available",
                    "confidence": 0.96,
                    "source": "verified-demo-service",
                },
                model="demo-backup",
            )
        ],
    )
    llm = ResilientLLM(
        RecoveryPlan([Route("primary", primary), Route("backup", backup)]),
        result_policy=validate_structured_result,
        failover_on_invalid_result=True,
    )

    response = llm.chat([{"role": "user", "content": "Give me an answer"}])
    print_result(response)


if __name__ == "__main__":
    main()
