# llm-api-resilience

`llm-api-resilience` is a Python resilience layer for multi-provider LLM
applications, built on top of
[`llm-api-adapter`](https://github.com/Inozem/llm_api_adapter).

The `v0.5.0` feature set adds route-specific prompt profiles, result-level
validation with opt-in failover, and declarative capability-aware routing on
top of retry, ordered failover, circuit breakers, application-managed
tool-calling sessions, provider-neutral checkpoints, and cross-provider
replay.

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

## Prompt profiles

Attach different system/developer instructions to individual routes. The
profile is applied at the request boundary, so the original messages and
session checkpoint remain unchanged:

```python
from llm_api_resilience import PromptProfile

llm = ResilientLLM(
    RecoveryPlan(
        [
            Route(
                "primary",
                primary_adapter,
                prompt_profile=PromptProfile(
                    system="Answer as a concise expert.",
                    developer="Prefer plain language.",
                ),
            ),
            Route(
                "backup",
                backup_adapter,
                prompt_profile=PromptProfile(
                    system="Answer as a helpful fallback assistant.",
                ),
            ),
        ]
    )
)
```

Profiles are immutable and are applied once per route attempt. During
cross-route session replay, the target route's profile is used. The current
`llm-api-adapter` boundary represents developer instructions as a labeled
section of one system message so the same profile works across providers.

## Result policies and result-level failover

The adapter already supports structural JSON handling through `json_schema`,
`response_model`, and `parsed_json`. `ResultPolicy` adds a separate,
application-level check for a response that was technically successful but is
not good enough for the business logic. For example, the JSON can be valid
while its confidence is too low or its source is not trusted:

```python
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_resilience import ResultDecision


def quality_policy(response: ChatResponse) -> ResultDecision:
    payload = response.parsed_json or {}
    valid = (
        isinstance(payload.get("answer"), str)
        and payload.get("confidence", 0) >= 0.8
        and payload.get("source") == "verified-service"
    )
    return ResultDecision(
        valid=valid,
        reason_type="semantic_quality_threshold",
    )


llm = ResilientLLM(
    recovery_plan,
    result_policy=quality_policy,
    failover_on_invalid_result=True,
)
response = llm.chat(messages)
```

With `failover_on_invalid_result=True`, an invalid response is recorded as a
safe failed attempt and the next route is tried. The route's retry policy is
still respected. Without the opt-in flag, `InvalidResultError` is raised and
the backup route is not called. The policy never executes tools itself.

`JSONSchemaError` is not automatically converted into result-level failover:
the adapter can also use that error family for invalid schemas or client-side
configuration errors. Use an explicit `ResultPolicy` when the application
wants to treat a returned result as unacceptable.

## Capability-aware routing

Applications can declare route capabilities and request only routes that meet
the current requirement:

```python
from llm_api_resilience import (
    CapabilityRequirements,
    RouteCapabilities,
)

llm = ResilientLLM(
    RecoveryPlan(
        [
            Route(
                "text",
                text_adapter,
                capabilities=RouteCapabilities(vision=False),
            ),
            Route(
                "vision",
                vision_adapter,
                capabilities=RouteCapabilities(vision=True),
            ),
        ]
    )
)

response = llm.chat(
    messages,
    capability_requirements=CapabilityRequirements(vision=True),
)
```

Routes without capability metadata remain backward-compatible and are treated
as unrestricted. Routes with declared missing capabilities are skipped before
the adapter is called and produce a safe `CapabilitySkipEvent`. If every
declared route is incompatible, `NoCompatibleRouteError` is raised.

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

Responses also expose `events`, while `ResilientLLM.last_events` contains the
events from the latest operation. A session exposes its accumulated events
through `session.events`. Circuit events contain only route, provider, model,
state, event type, error type, timestamp, and cooldown metadata. Capability
skip events contain only route, provider, model, missing capabilities, and a
timestamp. These records never contain API keys, request bodies, raw
responses, or complete error messages.

## Circuit breakers

Add a breaker to a route when a provider should be temporarily removed from
rotation after repeated failures:

```python
from llm_api_resilience import CircuitBreaker, RecoveryPlan, ResilientLLM, Route

primary_breaker = CircuitBreaker(
    failure_threshold=3,
    cooldown_s=30,
)

llm = ResilientLLM(
    RecoveryPlan(
        [
            Route("primary", primary_adapter, breaker=primary_breaker),
            Route("backup", backup_adapter),
        ]
    )
)
```

The breaker moves through three states:

- `closed` - requests are allowed;
- `open` - requests are skipped during the cooldown;
- `half_open` - one probe request checks whether the route recovered.

A successful probe closes the breaker. A failed probe opens it again. Breaker
state is shared by all sessions using the same `ResilientLLM` instance. To
manually reset every configured route breaker:

```python
llm.recovery_plan.reset_breakers()
```

Only errors classified as retryable by the configured `FailureClassifier`
increment the breaker. Non-retryable errors are raised immediately and do not
open the circuit.

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

## Limitations in `v0.5.0`

This version covers synchronous `chat()` calls, application-managed tool
loops, in-memory circuit breakers, prompt profiles, result policies, and
capability-aware routing. It does not yet provide async execution, streaming,
distributed breaker state, thread-safe coordination, automatic provider
capability discovery, or automatic tool execution. Tool execution remains the
application's responsibility.

Circuit state is local to the process that owns the `ResilientLLM` instance.
If a service runs multiple workers, each worker can make an independent route
decision. A shared state backend and distributed half-open coordination are
planned for a future production-hardening version.

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

The repository also includes an offline circuit-breaker example that requires
no API keys or network access:

```bash
python examples/circuit_breaker_observability.py
```

It demonstrates an initial failover, skipping an open primary route, and a
successful half-open recovery probe.

The result-level failover demo is also fully offline:

```bash
python examples/invalid_result_failover.py
```

It uses two valid JSON responses with different business-quality metadata.
The primary response has low confidence, so `ResultPolicy` rejects it and the
backup route is selected. No API keys or network access are required.

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
