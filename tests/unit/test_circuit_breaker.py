import pytest

from llm_api_resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_breaker_starts_closed_and_allows_requests():
    breaker = CircuitBreaker()

    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0
    assert breaker.allow_request() is True


def test_threshold_opens_breaker_and_rejects_requests_during_cooldown():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=2, cooldown_s=10, clock=clock)

    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()

    snapshot = breaker.snapshot()
    assert snapshot.state is CircuitState.OPEN
    assert snapshot.failure_count == 2
    assert snapshot.cooldown_remaining_s == pytest.approx(10)
    assert breaker.allow_request() is False


def test_open_breaker_allows_one_probe_after_cooldown():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=10, clock=clock)
    breaker.record_failure()

    clock.advance(10)

    assert breaker.allow_request() is True
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow_request() is False


def test_successful_half_open_probe_closes_and_resets_breaker():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=10, clock=clock)
    breaker.record_failure()
    clock.advance(10)

    assert breaker.allow_request() is True
    breaker.record_success()

    snapshot = breaker.snapshot()
    assert snapshot.state is CircuitState.CLOSED
    assert snapshot.failure_count == 0
    assert snapshot.opened_at is None
    assert breaker.allow_request() is True


def test_failed_half_open_probe_reopens_breaker_and_restarts_cooldown():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=10, clock=clock)
    breaker.record_failure()
    clock.advance(10)

    assert breaker.allow_request() is True
    breaker.record_failure()

    assert breaker.state is CircuitState.OPEN
    assert breaker.allow_request() is False
    assert breaker.snapshot().cooldown_remaining_s == pytest.approx(10)


def test_ensure_request_allowed_raises_safe_error_for_open_breaker():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=30, clock=clock)
    breaker.record_failure()

    with pytest.raises(CircuitOpenError, match="circuit is open") as error:
        breaker.ensure_request_allowed()

    assert error.value.cooldown_remaining_s == pytest.approx(30)


def test_reset_returns_breaker_to_initial_state():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=30)
    breaker.record_failure()

    breaker.reset()

    assert breaker.snapshot().state is CircuitState.CLOSED
    assert breaker.failure_count == 0
    assert breaker.allow_request() is True


@pytest.mark.parametrize(
    "kwargs, error_type, message",
    [
        ({"failure_threshold": 0}, ValueError, "at least 1"),
        ({"failure_threshold": True}, TypeError, "must be an integer"),
        ({"cooldown_s": -1}, ValueError, "non-negative"),
        ({"cooldown_s": "10"}, TypeError, "non-negative number"),
        ({"clock": object()}, TypeError, "callable"),
    ],
)
def test_breaker_rejects_invalid_configuration(kwargs, error_type, message):
    with pytest.raises(error_type, match=message):
        CircuitBreaker(**kwargs)


def test_circuit_open_error_rejects_invalid_cooldown_metadata():
    with pytest.raises(ValueError, match="non-negative"):
        CircuitOpenError(cooldown_remaining_s=-1)
