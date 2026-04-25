"""Unit tests for KubeMedic environment helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from kubernetes.client import ApiException

from Kubemedic.models import KubemedicAction
from Kubemedic.server.env import KubeMedicEnv


def _pod(
    name: str,
    phase: str,
    *,
    reason: str | None = None,
    node_name: str = "node-a",
    restarts: int = 0,
    ready: bool | None = None,
):
    container_status = SimpleNamespace(
        ready=(phase == "Running" if ready is None else ready),
        restart_count=restarts,
        state=SimpleNamespace(waiting=SimpleNamespace(reason=reason) if reason else None, terminated=None, running=None),
        last_state=SimpleNamespace(terminated=None),
    )
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, namespace="challenge"),
        spec=SimpleNamespace(node_name=node_name, priority_class_name=None),
        status=SimpleNamespace(
            phase=phase,
            reason=None,
            container_statuses=[container_status],
            conditions=[],
        ),
    )


def _node(name: str, *, ready: bool = True, pressure: bool = False):
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name),
        status=SimpleNamespace(
            allocatable={"cpu": "4", "memory": "16Gi"},
            conditions=[
                SimpleNamespace(type="Ready", status="True" if ready else "False"),
                SimpleNamespace(type="MemoryPressure", status="True" if pressure else "False"),
            ],
        ),
    )


def test_guard_blocks_protected_namespace_mutation() -> None:
    env = KubeMedicEnv(clients=SimpleNamespace(), tool_executor=SimpleNamespace())
    allowed, reason = env._guard(
        KubemedicAction(
            tool="kubectl_patch_resources",
            args={"deployment_name": "api-gw", "namespace": "kube-system", "container_name": "api-gw"},
        )
    )

    assert allowed is False
    assert "protected namespace" in str(reason)


def test_action_normalizes_string_args_for_kubectl_get() -> None:
    action = KubemedicAction(tool="kubectl_get", args="pods")

    assert action.args == {"resource": "pods"}


def test_action_normalizes_json_string_args() -> None:
    action = KubemedicAction(tool="kubectl_get", args='{"resource": "pods"}')

    assert action.args == {"resource": "pods"}


def test_guard_blocks_force_delete_outside_challenge() -> None:
    env = KubeMedicEnv(clients=SimpleNamespace(), tool_executor=SimpleNamespace())
    allowed, reason = env._guard(
        KubemedicAction(
            tool="kubectl_delete_pod",
            args={"pod_name": "x", "namespace": "default", "force": True},
        )
    )

    assert allowed is False
    assert "Force-deleting pods outside the challenge namespace" in str(reason)


def test_guard_blocks_workload_delete_outside_challenge() -> None:
    env = KubeMedicEnv(clients=SimpleNamespace(), tool_executor=SimpleNamespace())
    allowed, reason = env._guard(
        KubemedicAction(
            tool="kubectl_delete_workload",
            args={"resource": "daemonset", "name": "log-flood", "namespace": "default"},
        )
    )

    assert allowed is False
    assert "Mutating protected namespace" in str(reason)


def test_terminal_reward_respects_priority_speed_and_penalty() -> None:
    env = KubeMedicEnv(clients=SimpleNamespace(), tool_executor=SimpleNamespace())
    env.t = 9
    env.disruptions = 0
    env._initial_pressure_nodes = {"node-a"}
    env._list_challenge_pods = lambda: [_pod("api-gw-123", "Running"), _pod("order-svc-123", "Pending")]  # type: ignore[method-assign]
    env._all_healthy = lambda: False  # type: ignore[method-assign]
    env._node_name_under_pressure = lambda node_name: False  # type: ignore[method-assign]

    reward = env._terminal_reward()

    assert reward == 95


def test_obs_includes_structured_pods_and_nodes() -> None:
    env = KubeMedicEnv(clients=SimpleNamespace(), tool_executor=SimpleNamespace())
    env.t = 3
    env.scenario = "KUBE-03"
    env._list_challenge_pods = lambda: [_pod("payment-svc-123", "Failed", reason="OOMKilled", restarts=4)]  # type: ignore[method-assign]
    env._list_nodes = lambda: [_node("node-a", ready=True, pressure=False)]  # type: ignore[method-assign]

    obs = env._obs()

    assert obs["t"] == 3
    assert obs["scenario"] == "KUBE-03"
    assert obs["pods"][0]["reason"] == "OOMKilled"
    assert obs["nodes"][0]["name"] == "node-a"


def test_crashloop_running_pod_is_not_treated_as_healthy() -> None:
    env = KubeMedicEnv(clients=SimpleNamespace(), tool_executor=SimpleNamespace())
    env._list_challenge_pods = lambda: [_pod("payment-svc-123", "Running", reason="CrashLoopBackOff", ready=False)]  # type: ignore[method-assign]

    assert env._running_pod_names() == set()
    assert env._all_healthy() is False


@pytest.mark.anyio
async def test_wait_for_namespace_deleted_polls_until_gone() -> None:
    class _Core:
        def __init__(self) -> None:
            self.read_calls = 0

        def read_namespace(self, name: str) -> SimpleNamespace:
            self.read_calls += 1
            if self.read_calls < 3:
                return SimpleNamespace(status=SimpleNamespace(phase="Terminating"))
            raise ApiException(status=404)

    sleep_calls: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    core = _Core()
    env = KubeMedicEnv(
        clients=SimpleNamespace(core=core),
        tool_executor=SimpleNamespace(),
        sleep=_sleep,
    )

    await env._wait_for_namespace_deleted(timeout_s=5, interval_s=0.1)

    assert core.read_calls == 3
    assert sleep_calls == [0.1, 0.1]


@pytest.mark.anyio
async def test_wait_for_namespace_active_polls_until_active() -> None:
    class _Core:
        def __init__(self) -> None:
            self.read_calls = 0

        def read_namespace(self, name: str) -> SimpleNamespace:
            self.read_calls += 1
            if self.read_calls == 1:
                raise ApiException(status=404)
            if self.read_calls == 2:
                return SimpleNamespace(status=SimpleNamespace(phase="Terminating"))
            return SimpleNamespace(status=SimpleNamespace(phase="Active"))

    sleep_calls: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    core = _Core()
    env = KubeMedicEnv(
        clients=SimpleNamespace(core=core),
        tool_executor=SimpleNamespace(),
        sleep=_sleep,
    )

    await env._wait_for_namespace_active(timeout_s=5, interval_s=0.1)

    assert core.read_calls == 3
    assert sleep_calls == [0.1, 0.1]
