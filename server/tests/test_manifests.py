"""Tests for packaged Kubernetes manifests."""

from __future__ import annotations

from pathlib import Path

from Kubemedic.server.manifests import load_base_workloads, load_manifest


def test_load_base_workloads_reads_expected_deployments() -> None:
    documents = load_base_workloads()

    assert len(documents) == 3
    assert [doc["metadata"]["name"] for doc in documents] == [
        "api-gw",
        "auth-svc",
        "order-svc",
    ]
    assert all(doc["kind"] == "Deployment" for doc in documents)


def test_load_manifest_reads_multi_doc_yaml(tmp_path: Path) -> None:
    manifest_path = tmp_path / "workloads.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "kind: ConfigMap",
                "metadata:",
                "  name: a",
                "---",
                "kind: Secret",
                "metadata:",
                "  name: b",
            ]
        ),
        encoding="utf-8",
    )

    documents = load_manifest(manifest_path)

    assert [doc["kind"] for doc in documents] == ["ConfigMap", "Secret"]
