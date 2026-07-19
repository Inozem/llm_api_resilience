import pytest

from llm_api_adapter.models.responses.chat_response import ChatResponse

from llm_api_resilience import (
    RecoveryPlan,
    ResilientChatResponse,
    ResilientLLM,
    Route,
    RoutePolicy,
)


class FakeAdapter:
    def __init__(self, *, provider="fake", model="fake-model", response=None, error=None):
        self.organization = provider
        self.model = model
        self.response = response or ChatResponse(content="ok", model=model)
        self.error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def make_llm(adapter, *, timeout_s=None, backup=None):
    routes = [Route("primary", adapter, RoutePolicy(timeout_s=timeout_s))]
    if backup is not None:
        routes.append(Route("backup", backup))
    return ResilientLLM(RecoveryPlan(routes))


def test_resilient_llm_delegates_to_first_route_and_returns_compatible_response():
    adapter = FakeAdapter()
    llm = make_llm(adapter)
    messages = [{"role": "user", "content": "hello"}]

    response = llm.chat(messages, temperature=0.2, tools=["tool"])

    assert isinstance(response, ResilientChatResponse)
    assert isinstance(response, ChatResponse)
    assert response.selected_route == "primary"
    assert response.attempts[0].success is True
    assert adapter.calls[0]["messages"] is messages
    assert adapter.calls[0]["temperature"] == 0.2
    assert adapter.calls[0]["tools"] == ["tool"]


def test_resilient_llm_forwards_chat_kwargs_without_mutating_original_kwargs():
    adapter = FakeAdapter()
    llm = make_llm(adapter)
    request_kwargs = {
        "tools": ["tool"],
        "tool_choice": "auto",
        "json_schema": {"type": "object"},
        "response_model": object,
        "previous_response": object(),
    }

    llm.chat([], **request_kwargs)

    assert adapter.calls[0]["tools"] is request_kwargs["tools"]
    assert adapter.calls[0]["tool_choice"] == "auto"
    assert adapter.calls[0]["json_schema"] is request_kwargs["json_schema"]
    assert adapter.calls[0]["response_model"] is object
    assert adapter.calls[0]["previous_response"] is request_kwargs["previous_response"]
    assert "timeout_s" not in request_kwargs


def test_route_timeout_is_added_only_when_user_did_not_provide_one():
    adapter = FakeAdapter()
    llm = make_llm(adapter, timeout_s=12.0)

    llm.chat([])
    assert adapter.calls[-1]["timeout_s"] == 12.0

    llm.chat([], timeout_s=3.0)
    assert adapter.calls[-1]["timeout_s"] == 3.0


def test_resilient_llm_reraises_original_adapter_error_and_records_failure():
    error = RuntimeError("adapter failed")
    adapter = FakeAdapter(error=error)
    llm = make_llm(adapter)

    with pytest.raises(RuntimeError) as raised:
        llm.chat([])

    assert raised.value is error
    assert len(llm.last_attempts) == 1
    assert llm.last_attempts[0].success is False
    assert llm.last_attempts[0].error_type == "RuntimeError"
    assert llm.last_attempts[0].error_message == "adapter failed"


def test_resilient_llm_does_not_call_second_route_in_v01():
    primary = FakeAdapter(error=TimeoutError("primary failed"))
    backup = FakeAdapter()
    llm = make_llm(primary, backup=backup)

    with pytest.raises(TimeoutError):
        llm.chat([])

    assert len(primary.calls) == 1
    assert backup.calls == []

