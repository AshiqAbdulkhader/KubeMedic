"""AKS-backed Kubernetes client bootstrap for KubeMedic."""

from __future__ import annotations

import base64
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import yaml
from azure.identity import ClientSecretCredential
from azure.mgmt.containerservice import ContainerServiceClient
from dotenv import load_dotenv
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client import ApiClient


DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class ClusterConfigError(RuntimeError):
    """Raised when cluster configuration is missing or malformed."""


class ClusterConnectionError(RuntimeError):
    """Raised when cluster credentials cannot be resolved."""


@dataclass(frozen=True)
class AksClusterSettings:
    """Configuration required to connect to an AKS cluster."""

    client_id: str
    client_secret: str
    tenant_id: str
    subscription_id: str
    cluster_name: str
    resource_group: str

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "AksClusterSettings":
        """Load AKS settings from the repository `.env` file and process env."""

        dotenv_path = Path(env_file) if env_file else DEFAULT_ENV_FILE
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)

        values: dict[str, str] = {}
        missing: list[str] = []

        for env_name in (
            "CLIENT_ID",
            "CLIENT_SECRET",
            "TENANT_ID",
            "SUBSCRIPTION_ID",
            "CLUSTER_NAME",
            "RESOURCE_GROUP",
        ):
            value = os.getenv(env_name, "").strip()
            if value:
                values[env_name] = value
            else:
                missing.append(env_name)

        if missing:
            missing_list = ", ".join(missing)
            raise ClusterConfigError(
                f"Missing required AKS environment variables: {missing_list}"
            )

        return cls(
            client_id=values["CLIENT_ID"],
            client_secret=values["CLIENT_SECRET"],
            tenant_id=values["TENANT_ID"],
            subscription_id=values["SUBSCRIPTION_ID"],
            cluster_name=values["CLUSTER_NAME"],
            resource_group=values["RESOURCE_GROUP"],
        )


@dataclass
class KubernetesClients:
    """Convenience bundle of Kubernetes API clients."""

    api_client: ApiClient
    core: k8s_client.CoreV1Api
    apps: k8s_client.AppsV1Api
    batch: k8s_client.BatchV1Api
    networking: k8s_client.NetworkingV1Api
    custom_objects: k8s_client.CustomObjectsApi


def _safe_close(resource: Any) -> None:
    """Best-effort close for Azure and Kubernetes clients."""

    close = getattr(resource, "close", None)
    if callable(close):
        close()


def _decode_kubeconfig_payload(encoded_value: str | bytes) -> dict[str, Any]:
    """Decode raw or base64 kubeconfig payloads returned by AKS."""

    raw_bytes = (
        encoded_value.encode("utf-8")
        if isinstance(encoded_value, str)
        else bytes(encoded_value)
    )

    if raw_bytes.lstrip().startswith(b"apiVersion:"):
        kubeconfig_yaml = raw_bytes.decode("utf-8")
    else:
        try:
            kubeconfig_yaml = base64.b64decode(raw_bytes).decode("utf-8")
        except Exception as exc:  # pragma: no cover - defensive for Azure payloads
            raise ClusterConnectionError(
                "AKS returned an invalid kubeconfig payload"
            ) from exc

    kubeconfig = yaml.safe_load(kubeconfig_yaml)
    if not isinstance(kubeconfig, dict):
        raise ClusterConnectionError("Decoded kubeconfig payload was not a mapping")

    return kubeconfig


def _token_scope_for_cluster(kubeconfig: dict[str, Any]) -> str:
    """Build the Azure token scope for the AKS API server."""

    try:
        auth_provider_config = kubeconfig["users"][0]["user"]["auth-provider"]["config"]
        apiserver_id = auth_provider_config["apiserver-id"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ClusterConnectionError(
            "AKS kubeconfig did not expose an API server audience for token minting"
        ) from exc

    return (
        apiserver_id
        if apiserver_id.endswith("/.default")
        else f"{apiserver_id}/.default"
    )


def _inject_service_principal_token(
    kubeconfig: dict[str, Any],
    credential: ClientSecretCredential,
) -> dict[str, Any]:
    """Replace AKS Azure auth-provider config with a bearer token."""

    token = credential.get_token(_token_scope_for_cluster(kubeconfig))
    tokenized_kubeconfig = deepcopy(kubeconfig)
    tokenized_kubeconfig["users"][0]["user"] = {"token": token.token}
    return tokenized_kubeconfig


def _fetch_user_kubeconfig(
    settings: AksClusterSettings,
    credential: ClientSecretCredential,
) -> dict[str, Any]:
    """Fetch cluster user credentials and mint a Kubernetes bearer token."""

    aks_client = ContainerServiceClient(
        credential=credential,
        subscription_id=settings.subscription_id,
    )

    try:
        credential_results = aks_client.managed_clusters.list_cluster_user_credentials(
            resource_group_name=settings.resource_group,
            resource_name=settings.cluster_name,
            format="azure",
        )
        kubeconfigs = getattr(credential_results, "kubeconfigs", None) or []
        if not kubeconfigs:
            raise ClusterConnectionError(
                "AKS did not return any kubeconfig entries for the cluster"
            )

        raw_value = getattr(kubeconfigs[0], "value", None)
        if not raw_value:
            raise ClusterConnectionError("AKS returned an empty kubeconfig payload")

        kubeconfig = _decode_kubeconfig_payload(raw_value)
        return _inject_service_principal_token(kubeconfig, credential)
    except ClusterConnectionError:
        raise
    except Exception as exc:
        # Preserve Azure SDK details (status code/reason) so container logs
        # immediately show whether this is auth, RBAC, or wrong cluster metadata.
        details = str(exc).strip() or exc.__class__.__name__
        raise ClusterConnectionError(
            f"Failed to retrieve AKS user credentials from Azure: {details}"
        ) from exc
    finally:
        _safe_close(aks_client)


def _fetch_cluster_kubeconfig(settings: AksClusterSettings) -> dict[str, Any]:
    """Resolve a Kubernetes client config for the configured AKS cluster."""

    credential = ClientSecretCredential(
        tenant_id=settings.tenant_id,
        client_id=settings.client_id,
        client_secret=settings.client_secret,
    )

    try:
        return _fetch_user_kubeconfig(settings, credential)
    finally:
        _safe_close(credential)


class AksClusterClientFactory:
    """Creates and caches Kubernetes API clients for the configured AKS cluster."""

    def __init__(self, settings: AksClusterSettings | None = None):
        self.settings = settings or AksClusterSettings.from_env()
        self._api_client: ApiClient | None = None
        self._lock = Lock()

    def api_client(self, refresh: bool = False) -> ApiClient:
        """Return a configured Kubernetes ApiClient."""

        with self._lock:
            if refresh and self._api_client is not None:
                _safe_close(self._api_client)
                self._api_client = None

            if self._api_client is None:
                kubeconfig = _fetch_cluster_kubeconfig(self.settings)
                self._api_client = k8s_config.new_client_from_config_dict(
                    config_dict=kubeconfig,
                    persist_config=False,
                )

            return self._api_client

    def clients(self, refresh: bool = False) -> KubernetesClients:
        """Return a bundle of Kubernetes APIs backed by a shared ApiClient."""

        api_client = self.api_client(refresh=refresh)
        return KubernetesClients(
            api_client=api_client,
            core=k8s_client.CoreV1Api(api_client),
            apps=k8s_client.AppsV1Api(api_client),
            batch=k8s_client.BatchV1Api(api_client),
            networking=k8s_client.NetworkingV1Api(api_client),
            custom_objects=k8s_client.CustomObjectsApi(api_client),
        )

    def close(self) -> None:
        """Close cached client resources."""

        with self._lock:
            if self._api_client is not None:
                _safe_close(self._api_client)
                self._api_client = None
