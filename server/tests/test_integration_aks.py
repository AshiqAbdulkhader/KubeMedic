"""Opt-in AKS smoke tests for real reset/step flows."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable

import pytest

from Kubemedic.models import KubemedicAction
from Kubemedic.server.env import KubeMedicEnv


RUN_AKS_TESTS = os.getenv("KUBEMEDIC_RUN_AKS_TESTS") == "1"
pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        not RUN_AKS_TESTS,
        reason="Set KUBEMEDIC_RUN_AKS_TESTS=1 to run live AKS smoke tests",
    ),
]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _wait_until(
    predicate: Callable[[], Any],
    *,
    timeout_s: float = 180.0,
    interval_s: float = 5.0,
) -> Any:
    deadline = time.monotonic() + timeout_s
    last_value: Any = None

    while time.monotonic() < deadline:
        last_value = predicate()
        if last_value:
            return last_value
        await asyncio.sleep(interval_s)

    raise AssertionError(f"Condition not met within {timeout_s:.0f}s; last value: {last_value!r}")


def _pod_with_prefix(env: KubeMedicEnv, prefix: str) -> Any | None:
    for pod in env._list_challenge_pods():
        if pod.metadata.name.startswith(prefix):
            return pod
    return None


async def _cleanup_namespace(env: KubeMedicEnv) -> None:
    try:
        env._delete_namespace_if_present()
        await asyncio.sleep(2)
    finally:
        env.close()


async def test_kube_03_smoke_fix_with_resource_patch() -> None:
    env = KubeMedicEnv()
    try:
        observation = await env.reset("KUBE-03")
        assert any(pod.name.startswith("payment-svc") for pod in observation.pods)

        payment_pod = await _wait_until(
            lambda: _pod_with_prefix(env, "payment-svc"),
            timeout_s=120,
        )
        assert payment_pod.status.phase in {"Pending", "Running", "Failed"}

        describe = await env.step(
            KubemedicAction(
                tool="kubectl_describe",
                args={
                    "resource": "pod",
                    "name": payment_pod.metadata.name,
                    "namespace": "challenge",
                },
            )
        )
        container_states = describe.metadata["tool_result"]["containers"]
        assert any(
            container["last_state"]["reason"] == "OOMKilled"
            or container["restart_count"] >= 1
            for container in container_states
        )

        await env.step(
            KubemedicAction(
                tool="kubectl_patch_resources",
                args={
                    "deployment_name": "payment-svc",
                    "namespace": "challenge",
                    "container_name": "payment-svc",
                    "limits_memory_mi": 512,
                },
            )
        )

        fixed_pod = await _wait_until(
            lambda: next(
                (
                    pod
                    for pod in env._list_challenge_pods()
                    if pod.metadata.name.startswith("payment-svc")
                    and pod.status.phase == "Running"
                ),
                None,
            ),
            timeout_s=180,
        )
        assert fixed_pod is not None
    finally:
        await _cleanup_namespace(env)


async def test_kube_04_smoke_fix_with_request_patch() -> None:
    env = KubeMedicEnv()
    try:
        observation = await env.reset("KUBE-04")
        assert any(pod.name.startswith("ml-inference") for pod in observation.pods)

        ml_pod = await _wait_until(
            lambda: _pod_with_prefix(env, "ml-inference"),
            timeout_s=120,
        )
        assert ml_pod.status.phase == "Pending"

        describe = await env.step(
            KubemedicAction(
                tool="kubectl_describe",
                args={
                    "resource": "pod",
                    "name": ml_pod.metadata.name,
                    "namespace": "challenge",
                },
            )
        )
        assert any(
            "Insufficient memory" in event["message"]
            or "Insufficient cpu" in event["message"]
            for event in describe.metadata["tool_result"]["events"]
        )

        await env.step(
            KubemedicAction(
                tool="kubectl_patch_resources",
                args={
                    "deployment_name": "ml-inference",
                    "namespace": "challenge",
                    "container_name": "ml-inference",
                    "requests_memory_mi": 512,
                    "requests_cpu_m": 500,
                },
            )
        )

        fixed_pod = await _wait_until(
            lambda: next(
                (
                    pod
                    for pod in env._list_challenge_pods()
                    if pod.metadata.name.startswith("ml-inference")
                    and pod.status.phase == "Running"
                ),
                None,
            ),
            timeout_s=180,
        )
        assert fixed_pod is not None
    finally:
        await _cleanup_namespace(env)
