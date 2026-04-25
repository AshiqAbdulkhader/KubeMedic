"""Manifest loaders for server-side Kubernetes resources."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


def _load_yaml_documents(manifest_text: str) -> list[dict[str, Any]]:
    """Parse a multi-document YAML manifest into Kubernetes objects."""

    documents: list[dict[str, Any]] = []
    for document in yaml.safe_load_all(manifest_text):
        if not document:
            continue
        if not isinstance(document, dict):
            raise ValueError("Manifest document must be a mapping")
        documents.append(document)
    return documents


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load Kubernetes objects from a YAML file on disk."""

    return _load_yaml_documents(Path(path).read_text(encoding="utf-8"))


def load_base_workloads() -> list[dict[str, Any]]:
    """Load the default KubeMedic challenge workloads."""

    manifest = files("Kubemedic.server").joinpath("k8s").joinpath("base-workloads.yaml")
    return _load_yaml_documents(manifest.read_text(encoding="utf-8"))
