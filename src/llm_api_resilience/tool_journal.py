"""Safe journal for tool executions and replay decisions."""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

from llm_api_adapter.models.tools import ToolCall

from .session import ToolResult


class ReplayPolicy(str, Enum):
    """Whether a recorded tool execution may be reused during replay."""

    REPLAYABLE = "replayable"
    SIDE_EFFECTING = "side_effecting"


@dataclass(frozen=True, repr=False)
class ToolExecutionRecord:
    """A safe, defensive record of one completed or failed tool execution."""

    tool_call_id: str
    tool_name: str
    _arguments: Dict[str, Any]
    result: ToolResult
    replay_policy: ReplayPolicy
    status: str = "completed"
    idempotency_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip():
            raise ValueError("tool_call_id must be a non-empty string")
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if not isinstance(self._arguments, Mapping):
            raise TypeError("arguments must be a mapping")
        if not isinstance(self.result, ToolResult):
            raise TypeError("result must be a ToolResult")
        if not isinstance(self.replay_policy, ReplayPolicy):
            raise TypeError("replay_policy must be a ReplayPolicy")
        if self.status not in ("completed", "failed"):
            raise ValueError("status must be 'completed' or 'failed'")
        if self.idempotency_key is not None:
            if not isinstance(self.idempotency_key, str):
                raise TypeError("idempotency_key must be a string or None")
            if not self.idempotency_key.strip():
                raise ValueError("idempotency_key must not be empty")

        object.__setattr__(self, "_arguments", deepcopy(dict(self._arguments)))

    @property
    def arguments(self) -> Dict[str, Any]:
        """Return a fresh copy of the recorded tool arguments."""

        return deepcopy(self._arguments)

    def __repr__(self) -> str:
        return (
            "ToolExecutionRecord("
            f"tool_call_id={self.tool_call_id!r}, "
            f"tool_name={self.tool_name!r}, "
            f"replay_policy={self.replay_policy.value!r}, "
            f"status={self.status!r}, "
            f"has_idempotency_key={self.idempotency_key is not None}"
            ")"
        )


class ToolExecutionJournal:
    """Store tool results and resolve safe results during a future replay."""

    def __init__(self) -> None:
        self._entries: List[ToolExecutionRecord] = []
        self._by_fingerprint: Dict[str, ToolExecutionRecord] = {}
        self._by_idempotency_key: Dict[str, ToolExecutionRecord] = {}

    @property
    def entries(self) -> Tuple[ToolExecutionRecord, ...]:
        """All recorded executions in insertion order."""

        return tuple(self._entries)

    def record(
        self,
        tool_call: ToolCall,
        result: ToolResult,
        *,
        replay_policy: ReplayPolicy = ReplayPolicy.REPLAYABLE,
        status: str = "completed",
        idempotency_key: Optional[str] = None,
    ) -> ToolExecutionRecord:
        """Record an execution or return its existing idempotent record."""

        self._validate_tool_call(tool_call)
        if not isinstance(result, ToolResult):
            raise TypeError("result must be a ToolResult")
        if not isinstance(replay_policy, ReplayPolicy):
            raise TypeError("replay_policy must be a ReplayPolicy")

        effective_key = idempotency_key or result.idempotency_key
        if replay_policy is ReplayPolicy.SIDE_EFFECTING and not effective_key:
            raise ValueError(
                "side-effecting tool execution requires an idempotency_key"
            )

        fingerprint = self._fingerprint(tool_call)
        existing = None
        if effective_key:
            existing = self._by_idempotency_key.get(effective_key)
        elif replay_policy is ReplayPolicy.REPLAYABLE:
            existing = self._by_fingerprint.get(fingerprint)

        if existing is not None:
            self._ensure_same_invocation(existing, tool_call, effective_key)
            if (
                existing.status != status
                or existing.result.content != result.content
            ):
                raise ValueError("tool execution record conflicts with an existing result")
            return existing

        entry = ToolExecutionRecord(
            tool_call_id=tool_call.call_id or tool_call.name,
            tool_name=tool_call.name,
            _arguments=tool_call.arguments,
            result=result,
            replay_policy=replay_policy,
            status=status,
            idempotency_key=effective_key,
        )
        self._entries.append(entry)
        if effective_key:
            self._by_idempotency_key[effective_key] = entry
        if replay_policy is ReplayPolicy.REPLAYABLE:
            self._by_fingerprint[fingerprint] = entry
        return entry

    def lookup(
        self,
        tool_call: ToolCall,
        *,
        idempotency_key: Optional[str] = None,
    ) -> Optional[ToolExecutionRecord]:
        """Find a compatible recorded execution without executing the tool."""

        self._validate_tool_call(tool_call)
        entry = None
        if idempotency_key:
            entry = self._by_idempotency_key.get(idempotency_key)
        else:
            entry = self._by_fingerprint.get(self._fingerprint(tool_call))

        if entry is None:
            return None
        self._ensure_same_invocation(entry, tool_call, idempotency_key)
        if entry.status != "completed":
            return None
        return entry

    def replay_result(
        self,
        tool_call: ToolCall,
        *,
        idempotency_key: Optional[str] = None,
    ) -> Optional[ToolResult]:
        """Return a saved result, if this invocation can be replayed safely."""

        entry = self.lookup(tool_call, idempotency_key=idempotency_key)
        if entry is None:
            return None
        return ToolResult(
            tool_call_id=tool_call.call_id or tool_call.name,
            content=entry.result.content,
            idempotency_key=entry.idempotency_key,
        )

    @staticmethod
    def _validate_tool_call(tool_call: ToolCall) -> None:
        if not isinstance(tool_call, ToolCall):
            raise TypeError("tool_call must be a ToolCall")
        if not isinstance(tool_call.name, str) or not tool_call.name.strip():
            raise ValueError("tool_call.name must be a non-empty string")
        if not isinstance(tool_call.arguments, Mapping):
            raise TypeError("tool_call.arguments must be a mapping")

    @staticmethod
    def _fingerprint(tool_call: ToolCall) -> str:
        payload = json.dumps(
            {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _ensure_same_invocation(
        entry: ToolExecutionRecord,
        tool_call: ToolCall,
        idempotency_key: Optional[str],
    ) -> None:
        if entry.tool_name != tool_call.name or entry.arguments != tool_call.arguments:
            if idempotency_key:
                raise ValueError("idempotency_key is associated with another tool invocation")
            raise ValueError("tool invocation conflicts with an existing result")

    def __repr__(self) -> str:
        return f"ToolExecutionJournal(entry_count={len(self._entries)})"
