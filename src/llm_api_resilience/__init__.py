"""Resilience primitives for multi-provider LLM applications."""

from .attempts import AdapterProtocol, AttemptRecord
from .classifiers import DefaultFailureClassifier, FailureClassifier
from .errors import FailoverExhaustedError
from .policies import RoutePolicy
from .responses import ResilientChatResponse
from .resilient_llm import ResilientLLM
from .routes import RecoveryPlan, Route

__version__ = "0.1.0"

__all__ = [
    "AdapterProtocol",
    "AttemptRecord",
    "DefaultFailureClassifier",
    "FailoverExhaustedError",
    "FailureClassifier",
    "RecoveryPlan",
    "Route",
    "RoutePolicy",
    "ResilientChatResponse",
    "ResilientLLM",
    "__version__",
]
