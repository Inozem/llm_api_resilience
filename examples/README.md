# SafeRoute examples

## Recommended judge demo

This command requires no API keys and makes no network requests:

```powershell
python example_scripts/00_judge_offline_demo.py
```

It shows a deterministic provider failure and automatic failover through the
real `llm-api-resilience` library.

## Examples

- `00_judge_offline_demo.py` — deterministic failover for judges; no API keys.
- `01_openai_api_safe_route.py` — live GPT request with an offline fallback.
- `02_safe_route_demo.py` — basic offline failover.
- `03_tool_session_failover_demo.py` — the strongest feature: a tool result is
  replayed across a provider failure without executing the side effect twice.
