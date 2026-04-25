"""Core KubeMedic environment orchestration."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable
from uuid import uuid4

from kubernetes.client import ApiException, V1DeleteOptions, V1Namespace
from kubernetes.utils import create_from_dict
from openenv.core.env_server.types import State

try:
    from ..models import KubemedicAction, KubemedicObservation
except ImportError:
    from models import KubemedicAction, KubemedicObservation
from .cluster import AksClusterClientFactory, KubernetesClients
from .faults import FaultInjectionResult, cleanup as cleanup_faults, inject as inject_faults
from .manifests import load_base_workloads
from .spec import (
    CHALLENGE_NAMESPACE,
    MAX_SCALE_REPLICAS,
    MAX_STEPS,
    MUTATING_TOOLS,
    POD_PRIORITY,
    PROTECTED_NAMESPACES,
    SCENARIO_ROOT_CAUSES,
)
from .tools import KubeToolExecutor, pod_reason, serialize_node_summary, serialize_pod_summary


SleepFn = Callable[[float], Awaitable[None]]


class KubeMedicEnv:
    """AKS-backed Kubernetes repair environment."""

    def __init__(
        self,
        *,
        cluster_factory: AksClusterClientFactory | None = None,
        clients: KubernetesClients | None = None,
        tool_executor: KubeToolExecutor | None = None,
        sleep: SleepFn = asyncio.sleep,
    ):
        self._cluster_factory = cluster_factory if clients is None else None
        self._clients = clients
        self._tool_executor = tool_executor
        self._sleep = sleep

        self.t = 0
        self.disruptions = 0
        self.scenario = "KUBE-03"
        self._fault_result: FaultInjectionResult | None = None
        self._state = State(episode_id=str(uuid4()), step_count=0, scenario=self.scenario, disruptions=self.disruptions)
        self._initial_pressure_nodes: set[str] = set()

    @property
    def clients(self) -> KubernetesClients:
        if self._clients is None:
            factory = self._cluster_factory or AksClusterClientFactory()
            self._cluster_factory = factory
            self._clients = factory.clients()
        return self._clients

    @property
    def tools(self) -> KubeToolExecutor:
        if self._tool_executor is None:
            self._tool_executor = KubeToolExecutor(self.clients)
        return self._tool_executor

    @property
    def state(self) -> State:
        return self._state

    async def reset(self, scenario: str = "KUBE-03") -> KubemedicObservation:
        await self._cleanup_previous_faults()

        self.scenario = scenario.upper()
        self._state = State(
            episode_id=str(uuid4()),
            step_count=0,
            scenario=self.scenario,
            disruptions=0,
        )

        self._delete_namespace_if_present()
        await self._sleep(8)
        await self._wait_for_namespace_deleted()
        self._create_challenge_namespace()
        await self._wait_for_namespace_active()
        self._apply_base_workloads()
        await self._sleep(10)
        self._fault_result = inject_faults(self.clients, self.scenario)
        await self._sleep(15)

        self.t = 0
        self.disruptions = 0
        self._state.step_count = 0
        self._state.scenario = self.scenario
        self._state.disruptions = self.disruptions
        self._initial_pressure_nodes = {
            node["name"]
            for node in self._obs()["nodes"]
            if self._node_under_pressure(node)
        }

        return KubemedicObservation(
            **self._obs(),
            reward=0.0,
            done=False,
            metadata={
                "scenario_root_cause": SCENARIO_ROOT_CAUSES.get(self.scenario),
                "info": {
                    "disruptions": self.disruptions,
                    "steps_taken": self.t,
                },
            },
        )

    async def step(self, action: KubemedicAction) -> KubemedicObservation:
        self.t += 1
        self._state.step_count = self.t

        allowed, reason = self._guard(action)
        if not allowed:
            return KubemedicObservation(
                **self._obs(),
                reward=-5.0,
                done=False,
                metadata={
                    "scenario_root_cause": SCENARIO_ROOT_CAUSES.get(self.scenario),
                    "blocked_reason": reason,
                    "info": {"disruptions": self.disruptions, "steps_taken": self.t},
                },
            )

        prev_running = self._running_pod_names()
        reward = 0.0
        targeted_phase = None

        if action.tool == "kubectl_delete_pod":
            targeted_phase = self.tools.pod_phase(
                pod_name=action.args["pod_name"],
                namespace=action.args.get("namespace", CHALLENGE_NAMESPACE),
            )
            if targeted_phase == "Running":
                reward -= 25.0

        tool_result = self.tools.dispatch(action.tool, **action.args)
        await self._sleep(4)

        current_running = self._running_pod_names()
        recovered = len(current_running - prev_running)
        disrupted = len(prev_running - current_running)

        reward += recovered * 10.0
        if disrupted:
            reward -= disrupted * 25.0
            self.disruptions += disrupted

        done = self.t >= MAX_STEPS or self._all_healthy()
        if done:
            await self._cleanup_previous_faults()
            reward += self._terminal_reward()

        self._state.disruptions = self.disruptions
        observation_data = self._obs()
        return KubemedicObservation(
            **observation_data,
            reward=reward,
            done=done,
            metadata={
                "scenario_root_cause": SCENARIO_ROOT_CAUSES.get(self.scenario),
                "tool_result": tool_result,
                "info": {
                    "recovered": recovered,
                    "disrupted": disrupted,
                    "disruptions": self.disruptions,
                    "steps_taken": self.t,
                    "targeted_phase": targeted_phase,
                },
            },
        )

    def _obs(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "scenario": self.scenario,
            "pods": [serialize_pod_summary(pod) for pod in self._list_challenge_pods()],
            "nodes": [serialize_node_summary(node) for node in self._list_nodes()],
        }

    def _guard(self, action: KubemedicAction) -> tuple[bool, str | None]:
        namespace = action.args.get("namespace")
        if action.tool == "kubectl_delete_pod":
            pod_namespace = action.args.get("namespace", CHALLENGE_NAMESPACE)
            if action.args.get("force") and pod_namespace != CHALLENGE_NAMESPACE:
                return False, "Force-deleting pods outside the challenge namespace is blocked"

        if action.tool in MUTATING_TOOLS and namespace and namespace in PROTECTED_NAMESPACES:
            return False, f"Mutating protected namespace '{namespace}' is not allowed"

        if action.tool == "kubectl_delete_node":
            return False, "Node deletion is never allowed"

        replicas = action.args.get("replicas")
        if replicas is not None and int(replicas) > MAX_SCALE_REPLICAS:
            return False, f"Scaling above {MAX_SCALE_REPLICAS} replicas is blocked"

        if action.tool in {
            "kubectl_patch_resources",
            "kubectl_patch_tolerations",
            "kubectl_delete_pod",
            "kubectl_delete_workload",
        } and namespace and namespace != CHALLENGE_NAMESPACE:
            return False, "Mutating tools may only target the challenge namespace"

        return True, None

    def _terminal_reward(self) -> float:
        reward = 0.0
        pods = self._list_challenge_pods()
        for pod in pods:
            if self._is_healthy_pod(pod):
                priority = self._get_priority(pod.metadata.name)
                reward += 20 if priority == "critical" else 10

        if self._all_healthy():
            reward += 30

        if self.disruptions == 0:
            reward += 50

        if self.t < 10:
            reward += 20
        elif self.t < 15:
            reward += 10

        reward += 15 * sum(
            1 for node_name in self._initial_pressure_nodes if not self._node_name_under_pressure(node_name)
        )

        reward -= 10 * sum(1 for pod in pods if not self._is_healthy_pod(pod))
        return reward

    def _count_running(self) -> int:
        return len(self._running_pod_names())

    def _running_pod_names(self) -> set[str]:
        return {
            pod.metadata.name
            for pod in self._list_challenge_pods()
            if self._is_healthy_pod(pod)
        }

    def _all_healthy(self) -> bool:
        pods = self._list_challenge_pods()
        return bool(pods) and all(self._is_healthy_pod(pod) for pod in pods)

    def _is_healthy_pod(self, pod: Any) -> bool:
        if getattr(pod.status, "phase", None) != "Running":
            return False

        container_statuses = list(getattr(pod.status, "container_statuses", None) or [])
        if not container_statuses:
            return False

        if any(not getattr(status, "ready", False) for status in container_statuses):
            return False

        return pod_reason(pod) in {None, "Running"}

    def _get_priority(self, pod_name: str) -> str:
        for workload_name, priority in POD_PRIORITY.items():
            if pod_name == workload_name or pod_name.startswith(f"{workload_name}-"):
                return priority
        return "medium"

    def _node_under_pressure(self, node: dict[str, Any]) -> bool:
        return any(
            condition["type"] in {"MemoryPressure", "DiskPressure"}
            and condition["status"] == "True"
            for condition in node["conditions"]
        )

    def _node_name_under_pressure(self, node_name: str) -> bool:
        for node in self._list_nodes():
            if node.metadata.name != node_name:
                continue
            for condition in getattr(node.status, "conditions", None) or []:
                if condition.type in {"MemoryPressure", "DiskPressure"} and condition.status == "True":
                    return True
        return False

    def _list_challenge_pods(self) -> list[Any]:
        return self.clients.core.list_namespaced_pod(CHALLENGE_NAMESPACE).items

    def _list_nodes(self) -> list[Any]:
        return self.clients.core.list_node().items

    def _apply_base_workloads(self) -> None:
        for obj in load_base_workloads():
            create_from_dict(self.clients.api_client, data=obj, namespace=CHALLENGE_NAMESPACE)

    def _delete_namespace_if_present(self) -> None:
        try:
            self.clients.core.delete_namespace(
                name=CHALLENGE_NAMESPACE,
                body=V1DeleteOptions(),
            )
        except ApiException as exc:
            if exc.status != 404:
                raise

    def _create_challenge_namespace(self) -> None:
        try:
            self.clients.core.create_namespace(
                V1Namespace(metadata={"name": CHALLENGE_NAMESPACE})
            )
        except ApiException as exc:
            if exc.status != 409:
                raise

    async def _wait_for_namespace_deleted(
        self,
        *,
        timeout_s: float = 180.0,
        interval_s: float = 2.0,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            try:
                self.clients.core.read_namespace(CHALLENGE_NAMESPACE)
            except ApiException as exc:
                if exc.status == 404:
                    return
                raise
            await self._sleep(interval_s)

        raise TimeoutError(
            f"Namespace {CHALLENGE_NAMESPACE!r} was still present after {timeout_s:.0f}s"
        )

    async def _wait_for_namespace_active(
        self,
        *,
        timeout_s: float = 60.0,
        interval_s: float = 1.0,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            try:
                namespace = self.clients.core.read_namespace(CHALLENGE_NAMESPACE)
            except ApiException as exc:
                if exc.status == 404:
                    await self._sleep(interval_s)
                    continue
                raise

            if getattr(namespace.status, "phase", None) == "Active":
                return
            await self._sleep(interval_s)

        raise TimeoutError(
            f"Namespace {CHALLENGE_NAMESPACE!r} did not become Active after {timeout_s:.0f}s"
        )

    async def _cleanup_previous_faults(self) -> None:
        if self._fault_result is None:
            return
        cleanup_faults(
            self.clients,
            self._fault_result.scenario,
            self._fault_result.cleanup,
        )
        self._fault_result = None

    def close(self) -> None:
        if self._fault_result is not None:
            cleanup_faults(
                self.clients,
                self._fault_result.scenario,
                self._fault_result.cleanup,
            )
            self._fault_result = None
        if self._cluster_factory is not None:
            self._cluster_factory.close()
