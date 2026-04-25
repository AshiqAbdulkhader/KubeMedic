"""Scenario-specific fault injectors for the KubeMedic environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kubernetes import client as k8s_client
from kubernetes.utils import create_from_dict
from kubernetes.utils.quantity import parse_quantity

from .cluster import KubernetesClients
from .spec import CHALLENGE_NAMESPACE, SCENARIO_ROOT_CAUSES, SUPPORTED_SCENARIOS


@dataclass
class FaultInjectionResult:
    """Metadata captured during scenario setup."""

    scenario: str
    true_root_cause: str
    cleanup: dict[str, Any] = field(default_factory=dict)


def _apply_objects(clients: KubernetesClients, objects: list[dict[str, Any]]) -> None:
    for obj in objects:
        create_from_dict(clients.api_client, data=obj, namespace=CHALLENGE_NAMESPACE)


def _first_ready_node(clients: KubernetesClients) -> Any:
    for node in clients.core.list_node().items:
        if not getattr(node.spec, "unschedulable", False):
            for condition in getattr(node.status, "conditions", None) or []:
                if condition.type == "Ready" and condition.status == "True":
                    return node
    raise RuntimeError("No schedulable Ready node available for KUBE-05 injection")


def _node_hostname(node: Any) -> str:
    return (node.metadata.labels or {}).get("kubernetes.io/hostname", node.metadata.name)


def _max_node_allocatable(clients: KubernetesClients) -> tuple[int, int]:
    max_cpu_m = 0
    max_memory_mi = 0

    for node in clients.core.list_node().items:
        allocatable = getattr(node.status, "allocatable", {}) or {}
        cpu_m = int(parse_quantity(allocatable.get("cpu", "0")) * 1000)
        memory_mi = int(parse_quantity(allocatable.get("memory", "0")) / (1024 * 1024))
        max_cpu_m = max(max_cpu_m, cpu_m)
        max_memory_mi = max(max_memory_mi, memory_mi)

    if max_cpu_m == 0 or max_memory_mi == 0:
        raise RuntimeError("Unable to determine node allocatable resources for KUBE-04")

    return max_cpu_m, max_memory_mi


def _inject_kube_01(clients: KubernetesClients) -> FaultInjectionResult:
    node = _first_ready_node(clients)
    hostname = _node_hostname(node)
    _apply_objects(
        clients,
        [
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": "memory-hog",
                    "namespace": CHALLENGE_NAMESPACE,
                    "labels": {"app": "memory-hog", "priority": "background"},
                },
                "spec": {
                    "nodeSelector": {"kubernetes.io/hostname": hostname},
                    "restartPolicy": "Always",
                    "containers": [
                        {
                            "name": "memory-hog",
                            "image": "polinux/stress",
                            "args": ["--vm", "1", "--vm-bytes", "3072M", "--vm-keep"],
                            "resources": {
                                "requests": {"memory": "3072Mi", "cpu": "100m"},
                                "limits": {"memory": "3584Mi", "cpu": "500m"},
                            },
                        }
                    ],
                },
            },
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": "batch-job",
                    "namespace": CHALLENGE_NAMESPACE,
                    "labels": {"app": "batch-job", "priority": "background"},
                },
                "spec": {
                    "nodeSelector": {"kubernetes.io/hostname": hostname},
                    "restartPolicy": "Always",
                    "containers": [
                        {
                            "name": "batch-job",
                            "image": "busybox:1.36",
                            "command": ["sh", "-c", "sleep 3600"],
                        }
                    ],
                },
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": "worker-svc",
                    "namespace": CHALLENGE_NAMESPACE,
                    "labels": {"app": "worker-svc", "priority": "high"},
                },
                "spec": {
                    "replicas": 1,
                    "selector": {"matchLabels": {"app": "worker-svc"}},
                    "template": {
                        "metadata": {"labels": {"app": "worker-svc"}},
                        "spec": {
                            "nodeSelector": {"kubernetes.io/hostname": hostname},
                            "containers": [
                                {
                                    "name": "worker-svc",
                                    "image": "busybox:1.36",
                                    "command": ["sh", "-c", "sleep 3600"],
                                    "resources": {
                                        "requests": {"memory": "64Mi", "cpu": "50m"},
                                        "limits": {"memory": "256Mi", "cpu": "200m"},
                                    },
                                }
                            ],
                        },
                    },
                },
            },
        ],
    )
    return FaultInjectionResult(
        scenario="KUBE-01",
        true_root_cause=SCENARIO_ROOT_CAUSES["KUBE-01"],
        cleanup={"target_node": node.metadata.name},
    )


def _inject_kube_03(clients: KubernetesClients) -> FaultInjectionResult:
    _apply_objects(
        clients,
        [
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": "payment-svc",
                    "namespace": CHALLENGE_NAMESPACE,
                    "labels": {"app": "payment-svc", "priority": "critical"},
                },
                "spec": {
                    "replicas": 1,
                    "selector": {"matchLabels": {"app": "payment-svc"}},
                    "template": {
                        "metadata": {"labels": {"app": "payment-svc"}},
                        "spec": {
                            "containers": [
                                {
                                    "name": "payment-svc",
                                    "image": "polinux/stress",
                                    "args": ["--vm", "1", "--vm-bytes", "380M", "--vm-keep"],
                                    "resources": {
                                        "requests": {"memory": "128Mi", "cpu": "100m"},
                                        "limits": {"memory": "256Mi", "cpu": "250m"},
                                    },
                                }
                            ]
                        },
                    },
                },
            }
        ],
    )
    return FaultInjectionResult(
        scenario="KUBE-03",
        true_root_cause=SCENARIO_ROOT_CAUSES["KUBE-03"],
    )


def _inject_kube_04(clients: KubernetesClients) -> FaultInjectionResult:
    max_cpu_m, max_memory_mi = _max_node_allocatable(clients)
    impossible_cpu_m = max_cpu_m + 250
    impossible_memory_mi = max_memory_mi + 512

    _apply_objects(
        clients,
        [
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": "ml-inference",
                    "namespace": CHALLENGE_NAMESPACE,
                    "labels": {"app": "ml-inference", "priority": "medium"},
                },
                "spec": {
                    "replicas": 1,
                    "selector": {"matchLabels": {"app": "ml-inference"}},
                    "template": {
                        "metadata": {"labels": {"app": "ml-inference"}},
                        "spec": {
                            "containers": [
                                {
                                    "name": "ml-inference",
                                    "image": "nginx:alpine",
                                    "resources": {
                                        "requests": {
                                            "memory": f"{impossible_memory_mi}Mi",
                                            "cpu": f"{impossible_cpu_m}m",
                                        },
                                        "limits": {
                                            "memory": f"{impossible_memory_mi}Mi",
                                            "cpu": f"{impossible_cpu_m}m",
                                        },
                                    },
                                }
                            ]
                        },
                    },
                },
            }
        ],
    )
    return FaultInjectionResult(
        scenario="KUBE-04",
        true_root_cause=SCENARIO_ROOT_CAUSES["KUBE-04"],
    )


def _inject_kube_05(clients: KubernetesClients) -> FaultInjectionResult:
    node = _first_ready_node(clients)
    hostname = _node_hostname(node)
    taints = list(getattr(node.spec, "taints", None) or [])
    taints.append(
        k8s_client.V1Taint(
            key="gpu",
            value="true",
            effect="NoSchedule",
        )
    )
    clients.core.patch_node(node.metadata.name, {"spec": {"taints": [taint.to_dict() for taint in taints]}})

    _apply_objects(
        clients,
        [
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": "gpu-workload",
                    "namespace": CHALLENGE_NAMESPACE,
                    "labels": {"app": "gpu-workload", "priority": "medium"},
                },
                "spec": {
                    "replicas": 1,
                    "selector": {"matchLabels": {"app": "gpu-workload"}},
                    "template": {
                        "metadata": {"labels": {"app": "gpu-workload"}},
                        "spec": {
                            "nodeSelector": {"kubernetes.io/hostname": hostname},
                            "containers": [
                                {
                                    "name": "gpu-workload",
                                    "image": "nginx:alpine",
                                    "resources": {
                                        "requests": {"memory": "64Mi", "cpu": "50m"},
                                        "limits": {"memory": "128Mi", "cpu": "100m"},
                                    },
                                }
                            ],
                        },
                    },
                },
            }
        ],
    )
    return FaultInjectionResult(
        scenario="KUBE-05",
        true_root_cause=SCENARIO_ROOT_CAUSES["KUBE-05"],
        cleanup={"tainted_node": node.metadata.name},
    )


def _inject_kube_06(clients: KubernetesClients) -> FaultInjectionResult:
    _apply_objects(
        clients,
        [
            {
                "apiVersion": "apps/v1",
                "kind": "DaemonSet",
                "metadata": {
                    "name": "log-flood",
                    "namespace": CHALLENGE_NAMESPACE,
                    "labels": {"app": "log-flood", "priority": "background"},
                },
                "spec": {
                    "selector": {"matchLabels": {"app": "log-flood"}},
                    "template": {
                        "metadata": {"labels": {"app": "log-flood"}},
                        "spec": {
                            "terminationGracePeriodSeconds": 0,
                            "containers": [
                                {
                                    "name": "log-flood",
                                    "image": "busybox:1.36",
                                    "command": [
                                        "sh",
                                        "-c",
                                        (
                                            "mkdir -p /host-logs/kubemedic && "
                                            "while true; do "
                                            "head -c 1048576 /dev/zero >> /host-logs/kubemedic/app.log; "
                                            "sleep 1; "
                                            "done"
                                        ),
                                    ],
                                    "volumeMounts": [
                                        {"name": "host-logs", "mountPath": "/host-logs"}
                                    ],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "host-logs",
                                    "hostPath": {
                                        "path": "/var/log",
                                        "type": "Directory",
                                    },
                                }
                            ],
                        },
                    },
                },
            },
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": "batch-job",
                    "namespace": CHALLENGE_NAMESPACE,
                    "labels": {"app": "batch-job", "priority": "background"},
                },
                "spec": {
                    "restartPolicy": "Always",
                    "containers": [
                        {
                            "name": "batch-job",
                            "image": "busybox:1.36",
                            "command": ["sh", "-c", "sleep 3600"],
                        }
                    ],
                },
            },
        ],
    )
    return FaultInjectionResult(
        scenario="KUBE-06",
        true_root_cause=SCENARIO_ROOT_CAUSES["KUBE-06"],
    )


def inject(clients: KubernetesClients, scenario: str) -> FaultInjectionResult:
    """Apply the selected fault scenario to the challenge namespace."""

    scenario = scenario.upper()
    if scenario not in SUPPORTED_SCENARIOS:
        raise NotImplementedError(f"Scenario {scenario} is not implemented yet")
    if scenario == "KUBE-01":
        return _inject_kube_01(clients)
    if scenario == "KUBE-03":
        return _inject_kube_03(clients)
    if scenario == "KUBE-04":
        return _inject_kube_04(clients)
    if scenario == "KUBE-05":
        return _inject_kube_05(clients)
    if scenario == "KUBE-06":
        return _inject_kube_06(clients)
    raise NotImplementedError(f"Scenario {scenario} is not implemented yet")


def cleanup(clients: KubernetesClients, scenario: str | None, context: dict[str, Any] | None) -> None:
    """Revert scenario-level cluster mutations that outlive the namespace."""

    if not scenario or not context:
        return
    if scenario.upper() != "KUBE-05":
        return

    node_name = context.get("tainted_node")
    if not node_name:
        return

    node = clients.core.read_node(node_name)
    taints = list(getattr(node.spec, "taints", None) or [])
    filtered = [
        taint
        for taint in taints
        if not (
            getattr(taint, "key", None) == "gpu"
            and getattr(taint, "value", None) == "true"
            and getattr(taint, "effect", None) == "NoSchedule"
        )
    ]
    clients.core.patch_node(node_name, {"spec": {"taints": [taint.to_dict() for taint in filtered]}})
