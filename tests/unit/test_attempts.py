from dataclasses import asdict

import pytest

from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_resilience import AdapterProtocol, AttemptRecord

pytestmark = pytest.mark.unit


class SuccessfulFakeAdapter:
    def chat(self, **kwargs) -> ChatResponse:
        return ChatResponse(content="ok", model="fake-model")


class FailingFakeAdapter:
    def chat(self, **kwargs) -> ChatResponse:
        raise TimeoutError("request timed out")


def test_fake_adapter_satisfies_protocol_and_returns_chat_response():
    adapter = SuccessfulFakeAdapter()

    assert isinstance(adapter, AdapterProtocol)
    assert isinstance(adapter.chat(messages=[]), ChatResponse)


def test_failing_fake_adapter_can_be_used_without_network_access():
    adapter = FailingFakeAdapter()

    with pytest.raises(TimeoutError, match="timed out"):
        adapter.chat(messages=[])


def test_attempt_record_contains_route_and_duration():
    record = AttemptRecord(
        route_name="primary",
        provider="openai",
        model="gpt-test",
        duration_s=0.25,
        success=True,
    )

    assert record.route_name == "primary"
    assert record.provider == "openai"
    assert record.model == "gpt-test"
    assert record.duration_s == 0.25
    assert record.success is True


def test_attempt_record_stores_error_summary_without_secrets_or_request_body():
    record = AttemptRecord(
        route_name="primary",
        duration_s=1.5,
        success=False,
        error_type="TimeoutError",
        error_message="request timed out",
    )
    metadata = asdict(record)

    assert metadata["error_type"] == "TimeoutError"
    assert metadata["error_message"] == "request timed out"
    assert "api_key" not in metadata
    assert "request_body" not in metadata
