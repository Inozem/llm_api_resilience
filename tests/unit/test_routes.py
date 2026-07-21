import pytest

from llm_api_resilience import (
    CircuitBreaker,
    CircuitState,
    RecoveryPlan,
    Route,
    RoutePolicy,
)

pytestmark = pytest.mark.unit


class FakeAdapter:
    def chat(self, **kwargs):
        return kwargs


def make_route(name: str) -> Route:
    return Route(name=name, adapter=FakeAdapter())


def test_valid_route_uses_default_policy():
    route = make_route("primary")

    assert route.name == "primary"
    assert route.policy == RoutePolicy()
    assert route.breaker is None
    assert callable(route.adapter.chat)


def test_route_accepts_optional_circuit_breaker():
    breaker = CircuitBreaker(failure_threshold=2)

    route = Route(name="primary", adapter=FakeAdapter(), breaker=breaker)

    assert route.breaker is breaker


def test_route_rejects_invalid_circuit_breaker():
    with pytest.raises(TypeError, match="CircuitBreaker or None"):
        Route(name="primary", adapter=FakeAdapter(), breaker=object())


def test_route_rejects_empty_name():
    with pytest.raises(ValueError, match="must not be empty"):
        make_route("   ")


def test_route_rejects_adapter_without_chat():
    with pytest.raises(TypeError, match="callable chat"):
        Route(name="primary", adapter=object())


def test_recovery_plan_rejects_empty_routes():
    with pytest.raises(ValueError, match="at least one route"):
        RecoveryPlan(routes=[])


@pytest.mark.parametrize("routes", [None, "primary", b"primary"])
def test_recovery_plan_rejects_non_route_collections(routes):
    with pytest.raises(TypeError, match="iterable of Route objects"):
        RecoveryPlan(routes=routes)


def test_recovery_plan_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        RecoveryPlan(routes=[make_route("primary"), make_route("primary")])


def test_recovery_plan_rejects_shared_circuit_breaker():
    breaker = CircuitBreaker(failure_threshold=2)
    primary = Route(name="primary", adapter=FakeAdapter(), breaker=breaker)
    backup = Route(name="backup", adapter=FakeAdapter(), breaker=breaker)

    with pytest.raises(ValueError, match="own circuit breaker"):
        RecoveryPlan(routes=[primary, backup])


def test_recovery_plan_preserves_order_and_is_immutable():
    routes = [make_route("primary"), make_route("backup")]
    plan = RecoveryPlan(routes=routes)

    routes.append(make_route("later"))

    assert [route.name for route in plan] == ["primary", "backup"]
    assert isinstance(plan.routes, tuple)
    assert len(plan) == 2


def test_recovery_plan_resets_only_its_configured_route_breakers():
    primary_breaker = CircuitBreaker(failure_threshold=1)
    backup_breaker = CircuitBreaker(failure_threshold=1)
    primary = Route(
        name="primary",
        adapter=FakeAdapter(),
        breaker=primary_breaker,
    )
    backup = Route(
        name="backup",
        adapter=FakeAdapter(),
        breaker=backup_breaker,
    )
    outside_breaker = CircuitBreaker(failure_threshold=1)

    primary_breaker.record_failure()
    backup_breaker.record_failure()
    outside_breaker.record_failure()

    RecoveryPlan([primary, backup]).reset_breakers()

    assert primary_breaker.state is CircuitState.CLOSED
    assert backup_breaker.state is CircuitState.CLOSED
    assert outside_breaker.state is CircuitState.OPEN


def test_route_breakers_keep_independent_state():
    primary_breaker = CircuitBreaker(failure_threshold=1)
    backup_breaker = CircuitBreaker(failure_threshold=1)
    primary = Route(
        name="primary",
        adapter=FakeAdapter(),
        breaker=primary_breaker,
    )
    backup = Route(
        name="backup",
        adapter=FakeAdapter(),
        breaker=backup_breaker,
    )

    primary.breaker.record_failure()

    assert primary.breaker.state is CircuitState.OPEN
    assert backup.breaker.state is CircuitState.CLOSED


def test_route_policy_accepts_multiple_attempts():
    policy = RoutePolicy(max_attempts=3)

    assert policy.max_attempts == 3
    assert policy.backoff_s == 0.0
    assert policy.backoff_multiplier == 2.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_attempts": 0}, "at least 1"),
        ({"max_attempts": -1}, "at least 1"),
        ({"max_attempts": True}, "integer"),
        ({"backoff_s": -0.1}, "non-negative"),
        ({"backoff_s": True}, "non-negative number"),
        ({"backoff_multiplier": -1.0}, "non-negative"),
        ({"backoff_multiplier": "2"}, "non-negative number"),
    ],
)
def test_route_policy_rejects_invalid_retry_values(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        RoutePolicy(**kwargs)


def test_route_policy_calculates_exponential_backoff_in_order():
    policy = RoutePolicy(
        max_attempts=4,
        backoff_s=0.5,
        backoff_multiplier=2.0,
    )

    assert policy.backoff_for(1) == pytest.approx(0.5)
    assert policy.backoff_for(2) == pytest.approx(1.0)
    assert policy.backoff_for(3) == pytest.approx(2.0)
    assert policy.backoff_for(4) == 0.0


def test_route_policy_rejects_invalid_failed_attempt_number():
    policy = RoutePolicy(max_attempts=2)

    with pytest.raises(ValueError, match="at least 1"):
        policy.backoff_for(0)
    with pytest.raises(TypeError, match="must be an integer"):
        policy.backoff_for(1.5)

