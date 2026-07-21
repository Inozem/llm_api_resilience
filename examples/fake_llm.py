"""Fake LLM providers used by the local examples.

This module contains all offline provider behavior for the demos: scripted
responses and simulated provider errors. It never makes network requests.
"""

from typing import Any, List, Tuple

from llm_api_adapter.errors import LLMAPITimeoutError
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools import ToolCall


class ScriptedAdapter:
    """Small adapter double that returns predefined outcomes."""

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


def build_safe_route_adapters() -> Tuple[ScriptedAdapter, ScriptedAdapter]:
    """Build the fake primary and backup providers for the SafeRoute demo."""

    primary = ScriptedAdapter(
        organization="primary-provider",
        model="primary-model",
        outcomes=[LLMAPITimeoutError(detail="simulated provider timeout")],
    )
    return primary, build_safe_route_backup()


def build_safe_route_backup() -> ScriptedAdapter:
    """Build the offline backup used by the live API demo."""

    return ScriptedAdapter(
        organization="backup-provider",
        model="backup-model",
        outcomes=[
            ChatResponse(
                content="Инцидент обработан безопасным резервным маршрутом.",
                model="backup-model",
            )
        ],
    )


def build_tool_failover_adapters() -> Tuple[ScriptedAdapter, ScriptedAdapter]:
    """Build fake providers for a cross-provider tool-session replay demo."""

    primary = ScriptedAdapter(
        organization="primary-provider",
        model="primary-model",
        outcomes=[
            ChatResponse(
                model="primary-model",
                tool_calls=[
                    ToolCall(
                        name="create_incident_ticket",
                        arguments={"incident_id": "INC-42", "severity": "high"},
                        call_id="primary-call-1",
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMAPITimeoutError(detail="simulated continuation timeout"),
        ],
    )
    backup = ScriptedAdapter(
        organization="backup-provider",
        model="backup-model",
        outcomes=[
            ChatResponse(
                model="backup-model",
                tool_calls=[
                    ToolCall(
                        name="create_incident_ticket",
                        arguments={"incident_id": "INC-42", "severity": "high"},
                        call_id="backup-call-1",
                    )
                ],
                finish_reason="tool_calls",
            ),
            ChatResponse(
                content=(
                    "Incident ticket INC-42 was created successfully. "
                    "The operation was executed once and safely replayed."
                ),
                model="backup-model",
            ),
        ],
    )
    return primary, backup
