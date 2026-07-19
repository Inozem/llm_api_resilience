"""Resilience primitives for multi-provider LLM applications."""

from .attempts import AdapterProtocol, AttemptRecord
from .classifiers import DefaultFailureClassifier, FailureClassifier
from .checkpoints import Checkpoint, RouteIdentity
from .errors import FailoverExhaustedError, SessionStateError
from .policies import RoutePolicy
from .responses import ResilientChatResponse
from .resilient_llm import ResilientLLM
from .routes import RecoveryPlan, Route
from .session import ResilientSession, ToolResult

__version__ = "0.1.0"

__all__ = [
    "AdapterProtocol",
    "AttemptRecord",
    "DefaultFailureClassifier",
    "FailoverExhaustedError",
    "FailureClassifier",
    "Checkpoint",
    "RecoveryPlan",
    "Route",
    "RouteIdentity",
    "RoutePolicy",
    "ResilientSession",
    "ResilientChatResponse",
    "ResilientLLM",
    "SessionStateError",
    "ToolResult",
    "__version__",
]
