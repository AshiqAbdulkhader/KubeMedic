"""Tests for the AKS cluster bootstrap helpers."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from Kubemedic.server.cluster import (
    AksClusterClientFactory,
    AksClusterSettings,
    ClusterConfigError,
    _decode_kubeconfig_payload,
    _inject_service_principal_token,
    _token_scope_for_cluster,
)


def test_settings_load_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                'CLIENT_ID="client-id"',
                'CLIENT_SECRET="client-secret"',
                'TENANT_ID="tenant-id"',
                'SUBSCRIPTION_ID="subscription-id"',
                'CLUSTER_NAME="cluster-name"',
                'RESOURCE_GROUP="resource-group"',
            ]
        )
    )

    with patch.dict(os.environ, {}, clear=True):
        settings = AksClusterSettings.from_env(env_file)

    assert settings.client_id == "client-id"
    assert settings.cluster_name == "cluster-name"


def test_settings_raise_for_missing_required_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('CLIENT_ID="only-one-value"\n')

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ClusterConfigError):
            AksClusterSettings.from_env(env_file)


def test_decode_kubeconfig_payload_returns_mapping() -> None:
    encoded = base64.b64encode(
        b"apiVersion: v1\nclusters: []\ncontexts: []\nusers: []\n"
    )

    payload = _decode_kubeconfig_payload(encoded)

    assert payload["apiVersion"] == "v1"


def test_decode_kubeconfig_payload_handles_raw_yaml_bytes() -> None:
    payload = _decode_kubeconfig_payload(
        b"apiVersion: v1\nclusters: []\ncontexts: []\nusers: []\n"
    )

    assert payload["apiVersion"] == "v1"


def test_token_scope_uses_apiserver_id() -> None:
    scope = _token_scope_for_cluster(
        {
            "users": [
                {
                    "user": {
                        "auth-provider": {
                            "config": {"apiserver-id": "00000000-0000-0000-0000-000000000000"}
                        }
                    }
                }
            ]
        }
    )

    assert scope.endswith("/.default")


def test_inject_service_principal_token_rewrites_user_entry() -> None:
    kubeconfig = {
        "users": [
            {
                "name": "cluster-user",
                "user": {
                    "auth-provider": {
                        "config": {"apiserver-id": "00000000-0000-0000-0000-000000000000"}
                    }
                },
            }
        ]
    }

    fake_credential = type(
        "FakeCredential",
        (),
        {"get_token": lambda self, scope: type("Token", (), {"token": f"token-for:{scope}"})()},
    )()

    updated = _inject_service_principal_token(kubeconfig, fake_credential)

    assert updated["users"][0]["user"]["token"].startswith("token-for:")
    assert "auth-provider" in kubeconfig["users"][0]["user"]


def test_factory_builds_kubernetes_clients() -> None:
    settings = AksClusterSettings(
        client_id="client-id",
        client_secret="client-secret",
        tenant_id="tenant-id",
        subscription_id="subscription-id",
        cluster_name="cluster-name",
        resource_group="resource-group",
    )
    api_client = object()

    with patch(
        "Kubemedic.server.cluster._fetch_cluster_kubeconfig",
        return_value={"apiVersion": "v1"},
    ):
        with patch(
            "Kubemedic.server.cluster.k8s_config.new_client_from_config_dict",
            return_value=api_client,
        ):
            clients = AksClusterClientFactory(settings).clients()

    assert clients.api_client is api_client
