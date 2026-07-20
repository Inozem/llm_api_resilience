# llm-api-resilience

`llm-api-resilience` is a Python resilience layer for multi-provider LLM
applications, built on top of
[`llm-api-adapter`](https://github.com/Inozem/llm_api_adapter).

The `v0.3.0` feature set adds application-managed tool-calling sessions,
provider-neutral checkpoints, and safe cross-provider replay on top of retry
and ordered failover.

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

session = llm.session([{"role": "user", "content": "Hello"}])
response = session.start()
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

This example uses `ResilientSession` even though no tools are configured. A
text response completes the turn, and tools can be added later by passing a
`tools` list to the same session API. `llm.chat()` remains available for
simple one-shot requests.

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

## Chat sessions and optional tools

`ResilientSession` manages one application-driven chat turn. Tools are
optional: a text response completes the turn, while a tool call starts a
continuation. The application still executes tools; the resilience layer
stores the result, continues the conversation, and can replay the result if a
retryable continuation failure requires another provider.

```python
import json

from llm_api_resilience import ToolResult


session = llm.session(
    messages,
    tools=tools,
    tool_choice="auto",
    max_tokens=1000,
)

response = session.start()
while response.tool_calls:
    results = []
    for tool_call in response.tool_calls:
        value = run_tool(tool_call.name, tool_call.arguments)
        results.append(
            ToolResult(
                tool_call_id=tool_call.call_id or tool_call.name,
                content=json.dumps(value),
            )
        )
    # Send tool results back to the model. The response is either another
    # tool request or the final natural-language answer.
    response = session.continue_with(results)

final_response = response
print(final_response.content)
```

Before the first tool round, the session captures a provider-neutral
checkpoint. A same-route continuation can reuse `previous_response`. If that
continuation fails with a retryable error, the session restores the checkpoint,
tries the next route, and reuses completed tool results through its
`ToolExecutionJournal`.

For a side-effecting tool, provide an idempotency key so the result can be
replayed safely:

```python
ToolResult(
    tool_call_id=tool_call.call_id,
    content=json.dumps(value),
    idempotency_key="payment-123",
    replay_policy="side_effecting",
)
```

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

## Limitations in `v0.3.0`

This version covers synchronous `chat()` calls and application-managed tool
loops. It does not yet provide async execution, streaming, circuit breakers,
prompt profiles, result-based fallback, or automatic tool execution. Tool
execution remains the application's responsibility.

## E2E tests

The opt-in E2E test calls each provider through `ResilientLLM` and uses the
current model from `llm-api-adapter`'s model registry. Copy `.env.example` to
`.env`, add the provider keys, and run:

```bash
python -m pip install -e ".[test]"
pytest -m e2e
```

Cases without a configured key are skipped. The local `.env` file is ignored
by Git and must not contain committed credentials.

## Real multi-provider example

The repository includes a runnable example using real
`UniversalLLMAPIAdapter` instances for OpenAI, Anthropic, and Google:

```bash
python -m pip install -e ".[test]"
python examples/multi_provider_tool_failover.py
```

Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GOOGLE_API_KEY` in `.env`
before running it. The example executes a tool once and demonstrates how the
session continues and replays the result when a later route is required.

## License

MIT. See [LICENSE](LICENSE).
