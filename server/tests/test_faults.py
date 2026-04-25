"""Tests for scenario fault injectors."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from Kubemedic.server import faults


def _ready_node(name: str = "node-a") -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            labels={"kubernetes.io/hostname": f"{name}.internal"},
        ),
        spec=SimpleNamespace(taints=[]),
        status=SimpleNamespace(
            conditions=[SimpleNamespace(type="Ready", status="True")]
        ),
    )


def test_inject_kube_01_creates_memory_pressure_workloads() -> None:
    clients = SimpleNamespace()
    captured: list[dict[str, object]] = []

    with patch(
        "Kubemedic.server.faults._first_ready_node",
        return_value=_ready_node(),
    ):
        with patch(
            "Kubemedic.server.faults._apply_objects",
            side_effect=lambda _clients, objects: captured.extend(objects),
        ):
            result = faults.inject(clients, "KUBE-01")

    names = {obj["metadata"]["name"] for obj in captured}
    assert result.scenario == "KUBE-01"
    assert {"memory-hog", "batch-job", "worker-svc"} <= names


def test_inject_kube_06_creates_log_flood_daemonset() -> None:
    clients = SimpleNamespace()
    captured: list[dict[str, object]] = []

    with patch(
        "Kubemedic.server.faults._apply_objects",
        side_effect=lambda _clients, objects: captured.extend(objects),
    ):
        result = faults.inject(clients, "KUBE-06")

    kinds = {(obj["kind"], obj["metadata"]["name"]) for obj in captured}
    assert result.scenario == "KUBE-06"
    assert ("DaemonSet", "log-flood") in kinds
    assert ("Pod", "batch-job") in kinds


def test_inject_kube_04_requests_more_than_any_single_node() -> None:
    clients = SimpleNamespace()
    captured: list[dict[str, object]] = []

    with patch(
        "Kubemedic.server.faults._max_node_allocatable",
        return_value=(4000, 8192),
    ):
        with patch(
            "Kubemedic.server.faults._apply_objects",
            side_effect=lambda _clients, objects: captured.extend(objects),
        ):
            faults.inject(clients, "KUBE-04")

    container = captured[0]["spec"]["template"]["spec"]["containers"][0]
    assert container["resources"]["requests"] == {"memory": "8704Mi", "cpu": "4250m"}
    assert container["resources"]["limits"] == {"memory": "8704Mi", "cpu": "4250m"}


def test_cleanup_kube_05_removes_gpu_taint() -> None:
    node = SimpleNamespace(
        spec=SimpleNamespace(
            taints=[
                SimpleNamespace(key="gpu", value="true", effect="NoSchedule", to_dict=lambda: {"key": "gpu", "value": "true", "effect": "NoSchedule"}),
                SimpleNamespace(key="other", value="x", effect="NoSchedule", to_dict=lambda: {"key": "other", "value": "x", "effect": "NoSchedule"}),
            ]
        )
    )
    patched: dict[str, object] = {}
    clients = SimpleNamespace(
        core=SimpleNamespace(
            read_node=lambda _: node,
            patch_node=lambda node_name, patch: patched.update(
                {"node_name": node_name, "patch": patch}
            ),
        )
    )

    faults.cleanup(clients, "KUBE-05", {"tainted_node": "node-a"})

    assert patched["node_name"] == "node-a"
    assert patched["patch"] == {
        "spec": {
            "taints": [{"key": "other", "value": "x", "effect": "NoSchedule"}]
        }
    }
