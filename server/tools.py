"""Kubernetes-backed tool implementations for the KubeMedic environment."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from kubernetes import client as k8s_client
from kubernetes.client import ApiException
from kubernetes.utils.quantity import parse_quantity

from .cluster import KubernetesClients
from .spec import CHALLENGE_NAMESPACE


def _container_state_reason(container_status: Any) -> str | None:
    state = getattr(container_status, "state", None)
    if state is None:
        return None
    for attr in ("waiting", "terminated", "running"):
        value = getattr(state, attr, None)
        if value is not None:
            reason = getattr(value, "reason", None)
            if reason:
                return str(reason)
    return None


def pod_reason(pod: Any) -> str | None:
    """Infer the most useful reason for the pod's current condition."""

    if getattr(pod.status, "reason", None):
        return str(pod.status.reason)

    for status in getattr(pod.status, "container_statuses", None) or []:
        waiting_reason = _container_state_reason(status)
        if waiting_reason:
            return waiting_reason
        last_terminated = getattr(getattr(status, "last_state", None), "terminated", None)
        if last_terminated and getattr(last_terminated, "reason", None):
            return str(last_terminated.reason)

    for condition in getattr(pod.status, "conditions", None) or []:
        if getattr(condition, "status", None) == "False" and getattr(condition, "reason", None):
            return str(condition.reason)

    return None


def pod_restart_count(pod: Any) -> int:
    """Return the aggregate restart count across pod containers."""

    return sum(
        int(getattr(status, "restart_count", 0) or 0)
        for status in (getattr(pod.status, "container_statuses", None) or [])
    )


def serialize_pod_summary(pod: Any) -> dict[str, Any]:
    """Convert a Pod object into the observation summary shape."""

    return {
        "name": pod.metadata.name,
        "namespace": pod.metadata.namespace,
        "phase": pod.status.phase,
        "reason": pod_reason(pod),
        "node": getattr(pod.spec, "node_name", None),
        "restarts": pod_restart_count(pod),
        "priority_class": getattr(pod.spec, "priority_class_name", None),
    }


def _node_ready(node: Any) -> bool:
    for condition in getattr(node.status, "conditions", None) or []:
        if condition.type == "Ready":
            return condition.status == "True"
    return False


def serialize_node_summary(node: Any) -> dict[str, Any]:
    """Convert a Node object into the observation summary shape."""

    return {
        "name": node.metadata.name,
        "ready": _node_ready(node),
        "conditions": [
            {"type": condition.type, "status": condition.status}
            for condition in (getattr(node.status, "conditions", None) or [])
        ],
        "allocatable": {
            "cpu": (getattr(node.status, "allocatable", {}) or {}).get("cpu"),
            "memory": (getattr(node.status, "allocatable", {}) or {}).get("memory"),
        },
    }


def serialize_deployment(deployment: Any) -> dict[str, Any]:
    """Convert a Deployment object into a compact structured dict."""

    return {
        "name": deployment.metadata.name,
        "namespace": deployment.metadata.namespace,
        "replicas": deployment.spec.replicas,
        "ready_replicas": deployment.status.ready_replicas or 0,
        "available_replicas": deployment.status.available_replicas or 0,
        "labels": dict(deployment.metadata.labels or {}),
    }


def serialize_event(event: Any) -> dict[str, Any]:
    """Convert a Kubernetes event into a structured dict."""

    return {
        "reason": event.reason,
        "message": event.message,
        "count": event.count,
        "type": event.type,
        "involved_object": {
            "kind": getattr(event.involved_object, "kind", None),
            "name": getattr(event.involved_object, "name", None),
            "namespace": getattr(event.involved_object, "namespace", None),
        },
    }


def _timestamp_or_min(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.min.replace(tzinfo=timezone.utc)


def _parse_cpu_to_millicores(value: str | None) -> int | None:
    if not value:
        return None
    return int(parse_quantity(value) * 1000)


def _parse_memory_to_bytes(value: str | None) -> int | None:
    if not value:
        return None
    return int(parse_quantity(value))


class KubeToolExecutor:
    """Dispatcher for kubectl-like operations backed by the Kubernetes API."""

    def __init__(self, clients: KubernetesClients):
        self.clients = clients

    def dispatch(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        handler = getattr(self, tool_name, None)
        if handler is None:
            raise ValueError(f"Unsupported tool: {tool_name}")
        return handler(**kwargs)

    def _pod_events(self, namespace: str, name: str, limit: int = 8) -> list[dict[str, Any]]:
        events = self.clients.core.list_namespaced_event(
            namespace=namespace,
            field_selector=f"involvedObject.name={name}",
        ).items
        ordered = sorted(
            events,
            key=lambda event: _timestamp_or_min(
                getattr(event, "last_timestamp", None)
                or getattr(event, "event_time", None)
                or getattr(event, "first_timestamp", None)
                or getattr(event.metadata, "creation_timestamp", None)
            ),
        )
        return [serialize_event(event) for event in ordered[-limit:]]

    def _node_events(self, node_name: str, limit: int = 5) -> list[dict[str, Any]]:
        events = self.clients.core.list_event_for_all_namespaces(
            field_selector=f"involvedObject.kind=Node,involvedObject.name={node_name}",
        ).items
        ordered = sorted(
            events,
            key=lambda event: _timestamp_or_min(
                getattr(event, "last_timestamp", None)
                or getattr(event, "event_time", None)
                or getattr(event, "first_timestamp", None)
                or getattr(event.metadata, "creation_timestamp", None)
            ),
        )
        return [serialize_event(event) for event in ordered[-limit:]]

    def kubectl_get(
        self,
        resource: str,
        namespace: str = CHALLENGE_NAMESPACE,
        name: str | None = None,
    ) -> dict[str, Any]:
        resource = resource.lower()

        if resource == "pods":
            if name:
                pod = self.clients.core.read_namespaced_pod(name=name, namespace=namespace)
                return {"resource": resource, "namespace": namespace, "item": serialize_pod_summary(pod)}
            pods = self.clients.core.list_namespaced_pod(namespace=namespace).items
            return {"resource": resource, "namespace": namespace, "items": [serialize_pod_summary(pod) for pod in pods]}

        if resource == "nodes":
            if name:
                node = self.clients.core.read_node(name=name)
                return {"resource": resource, "item": serialize_node_summary(node)}
            nodes = self.clients.core.list_node().items
            return {"resource": resource, "items": [serialize_node_summary(node) for node in nodes]}

        if resource == "deployments":
            if name:
                deployment = self.clients.apps.read_namespaced_deployment(name=name, namespace=namespace)
                return {"resource": resource, "namespace": namespace, "item": serialize_deployment(deployment)}
            deployments = self.clients.apps.list_namespaced_deployment(namespace=namespace).items
            return {"resource": resource, "namespace": namespace, "items": [serialize_deployment(dep) for dep in deployments]}

        if resource == "events":
            events = self.clients.core.list_namespaced_event(namespace=namespace).items
            items = [serialize_event(event) for event in events]
            if name:
                items = [item for item in items if item["involved_object"]["name"] == name]
            return {"resource": resource, "namespace": namespace, "items": items}

        if resource == "pvc":
            if name:
                claim = self.clients.core.read_namespaced_persistent_volume_claim(name=name, namespace=namespace)
                return {
                    "resource": resource,
                    "namespace": namespace,
                    "item": {
                        "name": claim.metadata.name,
                        "status": claim.status.phase,
                        "volume_name": claim.spec.volume_name,
                        "storage_class": claim.spec.storage_class_name,
                    },
                }
            claims = self.clients.core.list_namespaced_persistent_volume_claim(namespace=namespace).items
            return {
                "resource": resource,
                "namespace": namespace,
                "items": [
                    {
                        "name": claim.metadata.name,
                        "status": claim.status.phase,
                        "volume_name": claim.spec.volume_name,
                        "storage_class": claim.spec.storage_class_name,
                    }
                    for claim in claims
                ],
            }

        if resource == "pv":
            if name:
                pv = self.clients.core.read_persistent_volume(name=name)
                return {
                    "resource": resource,
                    "item": {
                        "name": pv.metadata.name,
                        "status": pv.status.phase,
                        "storage_class": pv.spec.storage_class_name,
                        "claim_ref": getattr(pv.spec.claim_ref, "name", None),
                    },
                }
            pvs = self.clients.core.list_persistent_volume().items
            return {
                "resource": resource,
                "items": [
                    {
                        "name": pv.metadata.name,
                        "status": pv.status.phase,
                        "storage_class": pv.spec.storage_class_name,
                        "claim_ref": getattr(pv.spec.claim_ref, "name", None),
                    }
                    for pv in pvs
                ],
            }

        raise ValueError(f"Unsupported kubectl_get resource: {resource}")

    def kubectl_describe(
        self,
        resource: str,
        name: str,
        namespace: str = CHALLENGE_NAMESPACE,
    ) -> dict[str, Any]:
        resource = resource.lower()

        if resource in {"pod", "pods"}:
            pod = self.clients.core.read_namespaced_pod(name=name, namespace=namespace)
            return {
                "name": name,
                "phase": pod.status.phase,
                "conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": getattr(condition, "reason", None),
                        "message": getattr(condition, "message", None),
                    }
                    for condition in (getattr(pod.status, "conditions", None) or [])
                ],
                "containers": [
                    {
                        "name": status.name,
                        "image": next(
                            (
                                container.image
                                for container in (getattr(pod.spec, "containers", None) or [])
                                if container.name == status.name
                            ),
                            None,
                        ),
                        "ready": status.ready,
                        "restart_count": status.restart_count,
                        "last_state": {
                            "reason": getattr(
                                getattr(status.last_state, "terminated", None),
                                "reason",
                                None,
                            ),
                            "exit_code": getattr(
                                getattr(status.last_state, "terminated", None),
                                "exit_code",
                                None,
                            ),
                        },
                        "resources": {
                            "requests": dict(
                                (
                                    next(
                                        (
                                            container.resources.requests
                                            for container in (getattr(pod.spec, "containers", None) or [])
                                            if container.name == status.name
                                        ),
                                        {},
                                    )
                                    or {}
                                )
                            ),
                            "limits": dict(
                                (
                                    next(
                                        (
                                            container.resources.limits
                                            for container in (getattr(pod.spec, "containers", None) or [])
                                            if container.name == status.name
                                        ),
                                        {},
                                    )
                                    or {}
                                )
                            ),
                        },
                    }
                    for status in (getattr(pod.status, "container_statuses", None) or [])
                ],
                "events": self._pod_events(namespace, name, limit=8),
                "node": getattr(pod.spec, "node_name", None),
                "qos_class": getattr(pod.status, "qos_class", None),
            }

        if resource in {"node", "nodes"}:
            node = self.clients.core.read_node(name=name)
            return {
                "name": node.metadata.name,
                "conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": getattr(condition, "reason", None),
                        "message": getattr(condition, "message", None),
                    }
                    for condition in (getattr(node.status, "conditions", None) or [])
                ],
                "capacity": dict(getattr(node.status, "capacity", {}) or {}),
                "allocatable": dict(getattr(node.status, "allocatable", {}) or {}),
                "events": self._node_events(node.metadata.name, limit=5),
            }

        raise ValueError(f"Unsupported kubectl_describe resource: {resource}")

    def kubectl_logs(
        self,
        pod_name: str,
        namespace: str = CHALLENGE_NAMESPACE,
        previous: bool = False,
        tail: int = 50,
    ) -> dict[str, Any]:
        log_text = self.clients.core.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            previous=previous,
            tail_lines=tail,
        )
        return {
            "pod": pod_name,
            "namespace": namespace,
            "previous": previous,
            "lines": log_text.splitlines(),
        }

    def kubectl_top_pods(self, namespace: str = CHALLENGE_NAMESPACE) -> dict[str, Any]:
        metrics = self.clients.custom_objects.list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=namespace,
            plural="pods",
        )
        items = []
        for pod in metrics.get("items", []):
            containers = []
            cpu_total = 0
            memory_total = 0
            for container in pod.get("containers", []):
                cpu_m = _parse_cpu_to_millicores(container.get("usage", {}).get("cpu")) or 0
                memory_b = _parse_memory_to_bytes(container.get("usage", {}).get("memory")) or 0
                cpu_total += cpu_m
                memory_total += memory_b
                containers.append(
                    {
                        "name": container.get("name"),
                        "cpu_millicores": cpu_m,
                        "memory_bytes": memory_b,
                    }
                )
            items.append(
                {
                    "name": pod.get("metadata", {}).get("name"),
                    "cpu_millicores": cpu_total,
                    "memory_bytes": memory_total,
                    "containers": containers,
                }
            )
        return {"namespace": namespace, "items": items}

    def kubectl_top_nodes(self) -> dict[str, Any]:
        metrics = self.clients.custom_objects.list_cluster_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            plural="nodes",
        )
        items = []
        for node in metrics.get("items", []):
            usage = node.get("usage", {})
            items.append(
                {
                    "name": node.get("metadata", {}).get("name"),
                    "cpu_millicores": _parse_cpu_to_millicores(usage.get("cpu")),
                    "memory_bytes": _parse_memory_to_bytes(usage.get("memory")),
                }
            )
        return {"items": items}

    def kubectl_patch_resources(
        self,
        deployment_name: str,
        namespace: str,
        container_name: str,
        requests_memory_mi: int | None = None,
        limits_memory_mi: int | None = None,
        requests_cpu_m: int | None = None,
        limits_cpu_m: int | None = None,
    ) -> dict[str, Any]:
        container: dict[str, Any] = {"name": container_name, "resources": {}}
        if limits_memory_mi or limits_cpu_m:
            container["resources"]["limits"] = {}
            if limits_memory_mi:
                container["resources"]["limits"]["memory"] = f"{limits_memory_mi}Mi"
            if limits_cpu_m:
                container["resources"]["limits"]["cpu"] = f"{limits_cpu_m}m"
        if requests_memory_mi or requests_cpu_m:
            container["resources"]["requests"] = {}
            if requests_memory_mi:
                container["resources"]["requests"]["memory"] = f"{requests_memory_mi}Mi"
            if requests_cpu_m:
                container["resources"]["requests"]["cpu"] = f"{requests_cpu_m}m"

        patch = {"spec": {"template": {"spec": {"containers": [container]}}}}
        self.clients.apps.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=patch,
        )
        return {
            "deployment": deployment_name,
            "namespace": namespace,
            "patched_resources": container["resources"],
            "side_effects": [f"Patched resources on deployment/{deployment_name}"],
        }

    def kubectl_patch_tolerations(
        self,
        deployment_name: str,
        namespace: str,
        tolerations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        patch = {"spec": {"template": {"spec": {"tolerations": tolerations}}}}
        self.clients.apps.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=patch,
        )
        return {
            "deployment": deployment_name,
            "namespace": namespace,
            "tolerations": tolerations,
            "side_effects": [f"Patched tolerations on deployment/{deployment_name}"],
        }

    def kubectl_cordon(self, node_name: str) -> dict[str, Any]:
        patch = {"spec": {"unschedulable": True}}
        self.clients.core.patch_node(node_name, patch)
        return {
            "node": node_name,
            "action": "cordoned",
            "effect": "no new pods will be scheduled",
            "side_effects": [f"Marked node/{node_name} unschedulable"],
        }

    def kubectl_uncordon(self, node_name: str) -> dict[str, Any]:
        patch = {"spec": {"unschedulable": False}}
        self.clients.core.patch_node(node_name, patch)
        return {
            "node": node_name,
            "action": "uncordoned",
            "effect": "scheduling re-enabled",
            "side_effects": [f"Marked node/{node_name} schedulable"],
        }

    def kubectl_delete_pod(
        self,
        pod_name: str,
        namespace: str = CHALLENGE_NAMESPACE,
        force: bool = False,
    ) -> dict[str, Any]:
        if force:
            body = k8s_client.V1DeleteOptions(grace_period_seconds=0)
        else:
            body = k8s_client.V1DeleteOptions()
        self.clients.core.delete_namespaced_pod(
            name=pod_name,
            namespace=namespace,
            body=body,
        )
        return {
            "deleted": pod_name,
            "namespace": namespace,
            "force": force,
            "warning": "use only on Evicted/Failed pods",
            "side_effects": [f"Deleted pod/{pod_name}"],
        }

    def kubectl_delete_workload(
        self,
        resource: str,
        name: str,
        namespace: str = CHALLENGE_NAMESPACE,
    ) -> dict[str, Any]:
        resource = resource.lower()

        if resource in {"daemonset", "daemonsets"}:
            self.clients.apps.delete_namespaced_daemon_set(
                name=name,
                namespace=namespace,
                body=k8s_client.V1DeleteOptions(),
            )
            resource_name = "daemonset"
        elif resource in {"deployment", "deployments"}:
            self.clients.apps.delete_namespaced_deployment(
                name=name,
                namespace=namespace,
                body=k8s_client.V1DeleteOptions(),
            )
            resource_name = "deployment"
        else:
            raise ValueError(
                "kubectl_delete_workload supports only deployment and daemonset"
            )

        return {
            "resource": resource_name,
            "name": name,
            "namespace": namespace,
            "side_effects": [f"Deleted {resource_name}/{name}"],
        }

    def pod_phase(self, pod_name: str, namespace: str = CHALLENGE_NAMESPACE) -> str | None:
        try:
            pod = self.clients.core.read_namespaced_pod(name=pod_name, namespace=namespace)
        except ApiException as exc:
            if exc.status == 404:
                return None
            raise
        return pod.status.phase
