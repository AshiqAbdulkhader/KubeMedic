"""Targeted tests for tool helpers."""

from __future__ import annotations

from types import SimpleNamespace

from Kubemedic.server.tools import KubeToolExecutor, pod_reason, pod_restart_count


def test_pod_reason_prefers_container_reason() -> None:
    pod = SimpleNamespace(
        status=SimpleNamespace(
            reason=None,
            container_statuses=[
                SimpleNamespace(
                    restart_count=2,
                    state=SimpleNamespace(waiting=SimpleNamespace(reason="CrashLoopBackOff"), terminated=None, running=None),
                    last_state=SimpleNamespace(terminated=None),
                )
            ],
            conditions=[],
        )
    )

    assert pod_reason(pod) == "CrashLoopBackOff"
    assert pod_restart_count(pod) == 2


def test_patch_resources_builds_expected_patch() -> None:
    recorded: dict[str, object] = {}

    apps = SimpleNamespace(
        patch_namespaced_deployment=lambda name, namespace, body: recorded.update(
            {"name": name, "namespace": namespace, "body": body}
        )
    )
    clients = SimpleNamespace(apps=apps)
    executor = KubeToolExecutor(clients)

    result = executor.kubectl_patch_resources(
        deployment_name="payment-svc",
        namespace="challenge",
        container_name="payment-svc",
        limits_memory_mi=512,
    )

    assert recorded["name"] == "payment-svc"
    assert recorded["body"] == {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "payment-svc",
                            "resources": {"limits": {"memory": "512Mi"}},
                        }
                    ]
                }
            }
        }
    }
    assert "side_effects" in result


def test_delete_workload_routes_daemonset_delete() -> None:
    recorded: dict[str, object] = {}

    apps = SimpleNamespace(
        delete_namespaced_daemon_set=lambda name, namespace, body: recorded.update(
            {"name": name, "namespace": namespace, "body_type": type(body).__name__}
        )
    )
    clients = SimpleNamespace(apps=apps)
    executor = KubeToolExecutor(clients)

    result = executor.kubectl_delete_workload(
        resource="daemonset",
        name="log-flood",
        namespace="challenge",
    )

    assert recorded == {
        "name": "log-flood",
        "namespace": "challenge",
        "body_type": "V1DeleteOptions",
    }
    assert result["resource"] == "daemonset"
    assert "Deleted daemonset/log-flood" in result["side_effects"][0]
