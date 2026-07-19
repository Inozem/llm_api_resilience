# llm-api-resilience

`llm-api-resilience` is a Python resilience layer for multi-provider LLM
applications, built on top of
[`llm-api-adapter`](https://github.com/Inozem/llm_api_adapter).

The `v0.2.0` feature set adds retry and ordered failover between configured
models and providers while preserving the original `messages` and request
parameters.

## How failover works

Routes are tried in the order defined by `RecoveryPlan`. Each route has its
own `RoutePolicy`:

```python
from llm_api_resilience import RecoveryPlan, ResilientLLM, Route, RoutePolicy

llm = ResilientLLM(
    RecoveryPlan(
        [
            Route("primary", primary_adapter),
            Route(
                "backup",
                backup_adapter,
                RoutePolicy(
                    max_attempts=3,
                    backoff_s=0.25,
                    backoff_multiplier=2.0,
                ),
            ),
        ]
    )
)

response = llm.chat([{"role": "user", "content": "Hello"}])
```

`RoutePolicy()` performs one attempt, preserving the `v0.1` behavior. When a
retryable error occurs, the same route is retried up to `max_attempts`. After
the final attempt, the next route is tried. Backoff is applied only before a
next attempt; the delay sequence is `backoff_s`, then
`backoff_s * backoff_multiplier` and so on.

The default retryable errors are:

- `LLMAPITimeoutError`;
- `LLMAPIRateLimitError`;
- `LLMAPIServerError`.

Authorization, configuration, tool, JSON schema, invalid-input, and other
client-side errors are not retried or failed over by default.

For example, a request can move through `429 -> timeout -> success`:

```python
from llm_api_adapter.errors import LLMAPIRateLimitError, LLMAPITimeoutError
from llm_api_adapter.models.responses.chat_response import ChatResponse


class FakeAdapter:
    def __init__(self, outcome):
        self.outcome = outcome

    def chat(self, **kwargs):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


llm = ResilientLLM(
    RecoveryPlan(
        [
            Route("primary", FakeAdapter(LLMAPIRateLimitError())),
            Route("secondary", FakeAdapter(LLMAPITimeoutError())),
            Route(
                "fallback",
                FakeAdapter(ChatResponse(content="success", model="fallback")),
            ),
        ]
    )
)

response = llm.chat([{"role": "user", "content": "Hello"}])
assert response.selected_route == "fallback"
```

## Observability

Successful calls return a `ResilientChatResponse`, which remains compatible
with the adapter's `ChatResponse` and provides:

- `selected_route` — the route that returned the response;
- `attempts` — the complete attempt history for the operation;
- `ResilientLLM.last_attempts` — the latest operation history.

If all retryable attempts fail, `FailoverExhaustedError` is raised. It exposes
the aggregated `attempts` and the final `last_error`, which is also preserved
as the exception cause. Its message contains route, provider, model, and
error-type summaries without API keys or request bodies.

## Custom failure classification

Applications can provide their own classifier:

```python
class MyFailureClassifier:
    def is_retryable(self, error):
        return isinstance(error, (TimeoutError, ConnectionError))


llm = ResilientLLM(
    recovery_plan,
    failure_classifier=MyFailureClassifier(),
)
```

## Limitations in `v0.2.0`

This version covers synchronous ordinary `chat()` calls. It does not yet
provide async execution, streaming, circuit breakers, prompt profiles,
result-based fallback, automatic tool execution, tool checkpoints, or
cross-provider tool replay. Tool-call responses are returned as ordinary
responses without automatic execution or replay.

## License

MIT. See [LICENSE](LICENSE).
