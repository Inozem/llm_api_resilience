from datetime import datetime, timezone

import pytest

from llm_api_resilience import CircuitEvent, CircuitState

pytestmark = pytest.mark.unit


def test_circuit_event_contains_only_safe_route_metadata():
    timestamp = datetime.now(timezone.utc)
    event = CircuitEvent(
        event_type="opened",
        route_name="primary",
        state=CircuitState.OPEN,
        provider="openai",
        model="gpt-test",
        error_type="LLMAPITimeoutError",
        timestamp=timestamp,
        cooldown_remaining_s=30,
    )

    assert event.timestamp is timestamp
    assert event.state is CircuitState.OPEN
    assert event.error_type == "LLMAPITimeoutError"
    assert not hasattr(event, "error_message")
    assert "api_key" not in repr(event)
    assert "request_body" not in repr(event)


@pytest.mark.parametrize(
    "kwargs, error_type, message",
    [
        ({"event_type": "failed"}, ValueError, "event_type"),
        ({"route_name": ""}, ValueError, "route_name"),
        ({"state": "open"}, TypeError, "CircuitState"),
        ({"error_type": 1}, TypeError, "error_type"),
        ({"cooldown_remaining_s": -1}, ValueError, "non-negative"),
    ],
)
def test_circuit_event_rejects_invalid_metadata(kwargs, error_type, message):
    values = {
        "event_type": "opened",
        "route_name": "primary",
        "state": CircuitState.OPEN,
    }
    values.update(kwargs)

    with pytest.raises(error_type, match=message):
        CircuitEvent(**values)
