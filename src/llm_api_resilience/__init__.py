"""Resilience primitives for multi-provider LLM applications."""

from .attempts import AdapterProtocol, AttemptRecord
from .capabilities import (
    CapabilityRequirements,
    RouteCapabilities,
    capability_names,
)
from .classifiers import DefaultFailureClassifier, FailureClassifier
from .checkpoints import Checkpoint, RouteIdentity
from .circuit_breaker import CircuitBreaker, CircuitSnapshot, CircuitState
from .errors import (
    CapabilityMismatchError,
    CircuitOpenError,
    FailoverExhaustedError,
    InvalidResultError,
    NoCompatibleRouteError,
    SessionStateError,
)
from .observability import CapabilitySkipEvent, CircuitEvent
from .policies import RoutePolicy
from .prompt_profiles import PromptProfile
from .responses import ResilientChatResponse
from .result_policies import (
    ResultDecision,
    ResultPolicy,
    ResultPolicyCallback,
    evaluate_result_policy,
)
from .resilient_llm import ResilientLLM
from .routes import RecoveryPlan, Route
from .session import ResilientSession, ToolResult
from .tool_journal import ReplayPolicy, ToolExecutionJournal, ToolExecutionRecord

__version__ = "0.6.0"

__all__ = [
    "AdapterProtocol",
    "AttemptRecord",
    "CapabilityMismatchError",
    "CapabilityRequirements",
    "CapabilitySkipEvent",
    "CircuitBreaker",
    "CircuitEvent",
    "CircuitOpenError",
    "CircuitSnapshot",
    "CircuitState",
    "DefaultFailureClassifier",
    "FailoverExhaustedError",
    "FailureClassifier",
    "InvalidResultError",
    "NoCompatibleRouteError",
    "PromptProfile",
    "Checkpoint",
    "RecoveryPlan",
    "Route",
    "RouteIdentity",
    "RoutePolicy",
    "RouteCapabilities",
    "ResilientSession",
    "ResilientChatResponse",
    "ResilientLLM",
    "ResultDecision",
    "ResultPolicy",
    "ResultPolicyCallback",
    "evaluate_result_policy",
    "ReplayPolicy",
    "SessionStateError",
    "ToolExecutionJournal",
    "ToolExecutionRecord",
    "ToolResult",
    "capability_names",
    "__version__",
]
