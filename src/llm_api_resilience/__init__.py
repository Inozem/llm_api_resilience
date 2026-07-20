"""Resilience primitives for multi-provider LLM applications."""

from .attempts import AdapterProtocol, AttemptRecord
from .classifiers import DefaultFailureClassifier, FailureClassifier
from .checkpoints import Checkpoint, RouteIdentity
from .circuit_breaker import CircuitBreaker, CircuitSnapshot, CircuitState
from .errors import CircuitOpenError, FailoverExhaustedError, SessionStateError
from .observability import CircuitEvent
from .policies import RoutePolicy
from .prompt_profiles import PromptProfile
from .responses import ResilientChatResponse
from .resilient_llm import ResilientLLM
from .routes import RecoveryPlan, Route
from .session import ResilientSession, ToolResult
from .tool_journal import ReplayPolicy, ToolExecutionJournal, ToolExecutionRecord

__version__ = "0.1.0"

__all__ = [
    "AdapterProtocol",
    "AttemptRecord",
    "CircuitBreaker",
    "CircuitEvent",
    "CircuitOpenError",
    "CircuitSnapshot",
    "CircuitState",
    "DefaultFailureClassifier",
    "FailoverExhaustedError",
    "FailureClassifier",
    "PromptProfile",
    "Checkpoint",
    "RecoveryPlan",
    "Route",
    "RouteIdentity",
    "RoutePolicy",
    "ResilientSession",
    "ResilientChatResponse",
    "ResilientLLM",
    "ReplayPolicy",
    "SessionStateError",
    "ToolExecutionJournal",
    "ToolExecutionRecord",
    "ToolResult",
    "__version__",
]
