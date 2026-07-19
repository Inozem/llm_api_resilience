# llm-api-resilience

`llm-api-resilience` is a Python resilience layer for multi-provider LLM
applications, built on top of
[`llm-api-adapter`](https://github.com/Inozem/llm_api_adapter).

The library provides named routes, immutable recovery plans, transparent
`chat()` delegation, `ChatResponse`-compatible results, and safe attempt
metadata for provider, model, duration, success state, and errors.

The current `v0.1.0` release is the foundation of the project. It executes a
single configured route and records its result; retry, automatic provider
failover, checkpointed tool recovery, and circuit breakers are planned for
future releases.

## License

MIT. See [LICENSE](LICENSE).
