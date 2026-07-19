import pytest

from llm_api_resilience import RecoveryPlan, Route, RoutePolicy


class FakeAdapter:
    def chat(self, **kwargs):
        return kwargs


def make_route(name: str) -> Route:
    return Route(name=name, adapter=FakeAdapter())


def test_valid_route_uses_default_policy():
    route = make_route("primary")

    assert route.name == "primary"
    assert route.policy == RoutePolicy()
    assert callable(route.adapter.chat)


def test_route_rejects_empty_name():
    with pytest.raises(ValueError, match="must not be empty"):
        make_route("   ")


def test_route_rejects_adapter_without_chat():
    with pytest.raises(TypeError, match="callable chat"):
        Route(name="primary", adapter=object())


def test_recovery_plan_rejects_empty_routes():
    with pytest.raises(ValueError, match="at least one route"):
        RecoveryPlan(routes=[])


def test_recovery_plan_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        RecoveryPlan(routes=[make_route("primary"), make_route("primary")])


def test_recovery_plan_preserves_order_and_is_immutable():
    routes = [make_route("primary"), make_route("backup")]
    plan = RecoveryPlan(routes=routes)

    routes.append(make_route("later"))

    assert [route.name for route in plan] == ["primary", "backup"]
    assert isinstance(plan.routes, tuple)
    assert len(plan) == 2


def test_route_policy_rejects_retry_in_v01():
    with pytest.raises(ValueError, match="exactly one attempt"):
        RoutePolicy(max_attempts=2)

