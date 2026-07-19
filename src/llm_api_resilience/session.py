"""Provider-neutral sessions for application-managed tool loops."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterable, List, Optional, Tuple, TYPE_CHECKING

from llm_api_adapter.models.messages.chat_message import AIMessage, ToolMessage
from llm_api_adapter.models.responses.chat_response import ChatResponse

from .attempts import AttemptRecord
from .checkpoints import Checkpoint, RouteIdentity
from .errors import SessionStateError
from .responses import ResilientChatResponse
from .routes import Route

if TYPE_CHECKING:
    from .resilient_llm import ResilientLLM


@dataclass(frozen=True)
class ToolResult:
    """A provider-neutral result produced by the application for one tool call."""

    tool_call_id: str
    content: str
    idempotency_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_call_id, str):
            raise TypeError("tool_call_id must be a string")
        if not self.tool_call_id.strip():
            raise ValueError("tool_call_id must not be empty")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if self.idempotency_key is not None:
            if not isinstance(self.idempotency_key, str):
                raise TypeError("idempotency_key must be a string or None")
            if not self.idempotency_key.strip():
                raise ValueError("idempotency_key must not be empty")


class ResilientSession:
    """Manage an application-driven tool loop on the selected route.

    This initial session implementation supports a first tool call and
    same-route continuation. Retry, journal-backed replay, and cross-route
    recovery are added in later versions.
    """

    def __init__(
        self,
        llm: "ResilientLLM",
        messages: Iterable[Any],
        request_kwargs: dict,
    ) -> None:
        if not hasattr(llm, "chat") or not hasattr(llm, "recovery_plan"):
            raise TypeError("llm must be a ResilientLLM")
        if isinstance(messages, (str, bytes)):
            raise TypeError("messages must be an iterable of message objects")

        try:
            initial_messages = tuple(deepcopy(tuple(messages)))
        except TypeError as exc:
            raise TypeError("messages must be an iterable") from exc

        self._llm = llm
        self._initial_messages = initial_messages
        self._working_messages: List[Any] = list(deepcopy(initial_messages))
        self._request_kwargs = deepcopy(dict(request_kwargs))
        self._attempts: List[AttemptRecord] = []
        self._checkpoint: Optional[Checkpoint] = None
        self._active_route: Optional[Route] = None
        self._response: Optional[ResilientChatResponse] = None
        self._pending_tool_results: Optional[Tuple[ToolResult, ...]] = None
        self._started = False
        self._closed = False

    @property
    def checkpoint(self) -> Optional[Checkpoint]:
        """The snapshot captured before the first tool round, if any."""

        return self._checkpoint

    @property
    def attempts(self) -> Tuple[AttemptRecord, ...]:
        """All attempts made by this session so far."""

        return tuple(self._attempts)

    @property
    def response(self) -> Optional[ResilientChatResponse]:
        """The most recent response returned by the session."""

        return self._response

    @property
    def is_closed(self) -> bool:
        """Whether the session has returned a final response."""

        return self._closed

    def start(self) -> ResilientChatResponse:
        """Execute the first route call and capture a tool checkpoint if needed."""

        if self._started:
            raise SessionStateError("session.start() can only be called once")
        self._started = True

        response = self._llm.chat(
            self._working_messages,
            **deepcopy(self._request_kwargs),
        )
        self._response = response
        self._attempts = list(response.attempts)
        self._active_route = self._route_for_response(response)
        self._llm._last_attempts = tuple(self._attempts)

        if response.tool_calls:
            self._checkpoint = Checkpoint.capture(
                messages=self._initial_messages,
                request_kwargs=self._request_kwargs,
                route=self._route_identity(self._active_route),
            )
        else:
            self._closed = True

        return response

    def continue_with(
        self,
        tool_results: Iterable[ToolResult],
    ) -> ResilientChatResponse:
        """Continue the current route after the application executes tools."""

        self._ensure_started()
        if self._closed:
            raise SessionStateError("session already returned a final response")
        if self._response is None or not self._response.tool_calls:
            raise SessionStateError("current response does not contain tool calls")
        if self._active_route is None:
            raise SessionStateError("session has no active route")

        normalized_results = self._normalize_tool_results(tool_results)
        self._validate_tool_results(normalized_results)
        if self._pending_tool_results is None:
            self._append_tool_round(normalized_results)
            self._pending_tool_results = normalized_results
        elif normalized_results != self._pending_tool_results:
            raise SessionStateError(
                "retrying a continuation must use the same tool results"
            )

        request_kwargs = self._llm._build_request_kwargs(
            self._request_kwargs,
            route=self._active_route,
            include_previous_response=False,
        )
        current_route = self._route_identity(self._active_route)
        if (
            self._checkpoint is not None
            and self._checkpoint.route.is_compatible_with(current_route)
        ):
            request_kwargs["previous_response"] = self._response
        started_at = datetime.now(timezone.utc)
        started_tick = perf_counter()

        try:
            response = self._active_route.adapter.chat(
                messages=self._working_messages,
                **request_kwargs,
            )
        except Exception as error:
            attempt = self._llm._make_attempt_record(
                route=self._active_route,
                started_at=started_at,
                duration_s=perf_counter() - started_tick,
                success=False,
                error=error,
            )
            self._attempts.append(attempt)
            self._llm._last_attempts = tuple(self._attempts)
            raise

        attempt = self._llm._make_attempt_record(
            route=self._active_route,
            started_at=started_at,
            duration_s=perf_counter() - started_tick,
            success=True,
        )
        self._attempts.append(attempt)
        self._llm._last_attempts = tuple(self._attempts)
        self._response = ResilientChatResponse.from_chat_response(
            response,
            selected_route=self._active_route.name,
            attempts=tuple(self._attempts),
        )
        self._pending_tool_results = None
        if not response.tool_calls:
            self._closed = True
        return self._response

    def _ensure_started(self) -> None:
        if not self._started:
            raise SessionStateError("call session.start() before continuing")

    def _route_for_response(self, response: ChatResponse) -> Route:
        selected_route = response.selected_route
        for route in self._llm.recovery_plan:
            if route.name == selected_route:
                return route
        raise SessionStateError("response selected an unknown route")

    @staticmethod
    def _route_identity(route: Route) -> RouteIdentity:
        provider = getattr(route.adapter, "organization", None)
        if not isinstance(provider, str) or not provider:
            provider = getattr(route.adapter, "company", None)
        if not isinstance(provider, str) or not provider:
            provider = None
        model = getattr(route.adapter, "model", None)
        if not isinstance(model, str) or not model:
            model = None
        return RouteIdentity(route.name, provider, model)

    @staticmethod
    def _normalize_tool_results(
        tool_results: Iterable[ToolResult],
    ) -> Tuple[ToolResult, ...]:
        if isinstance(tool_results, ToolResult):
            return (tool_results,)
        try:
            normalized = tuple(tool_results)
        except TypeError as exc:
            raise TypeError("tool_results must be an iterable of ToolResult") from exc
        if any(not isinstance(result, ToolResult) for result in normalized):
            raise TypeError("tool_results must contain ToolResult objects")
        return normalized

    def _validate_tool_results(
        self,
        tool_results: Tuple[ToolResult, ...],
    ) -> None:
        expected_ids = tuple(
            call.call_id or call.name for call in self._response.tool_calls
        )
        actual_ids = tuple(result.tool_call_id for result in tool_results)
        if len(actual_ids) != len(set(actual_ids)):
            raise ValueError("tool_results must not contain duplicate tool_call_id values")
        if set(actual_ids) != set(expected_ids):
            raise ValueError("tool_results must match the current tool calls")

    def _append_tool_round(self, tool_results: Tuple[ToolResult, ...]) -> None:
        self._working_messages.append(
            AIMessage(
                content=self._response.content or "",
                tool_calls=list(self._response.tool_calls),
            )
        )
        self._working_messages.extend(
            ToolMessage(
                tool_call_id=result.tool_call_id,
                content=result.content,
            )
            for result in tool_results
        )
