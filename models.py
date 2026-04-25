# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client-facing data models for the Kubemedic environment."""

from __future__ import annotations

import json
from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, model_validator


ToolName = Literal[
    "kubectl_get",
    "kubectl_describe",
    "kubectl_logs",
    "kubectl_top_pods",
    "kubectl_top_nodes",
    "kubectl_patch_resources",
    "kubectl_patch_tolerations",
    "kubectl_cordon",
    "kubectl_uncordon",
    "kubectl_delete_pod",
    "kubectl_delete_workload",
]


class PodObservation(BaseModel):
    name: str
    namespace: str
    phase: str
    reason: str | None = None
    node: str | None = None
    restarts: int = 0
    priority_class: str | None = None


class NodeConditionObservation(BaseModel):
    type: str
    status: str


class NodeObservation(BaseModel):
    name: str
    ready: bool
    conditions: list[NodeConditionObservation] = Field(default_factory=list)
    allocatable: dict[str, Any] = Field(default_factory=dict)


class KubemedicAction(Action):
    """Tool invocation sent to the KubeMedic environment."""

    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_args(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        args = normalized.get("args")

        if args is None:
            normalized["args"] = {}
            return normalized

        if isinstance(args, str):
            stripped_args = args.strip()
            if stripped_args.startswith("{") and stripped_args.endswith("}"):
                parsed_args = json.loads(stripped_args)
                if not isinstance(parsed_args, dict):
                    raise TypeError("JSON args payload must decode to an object")
                normalized["args"] = parsed_args
                return normalized

            if normalized.get("tool") == "kubectl_get":
                normalized["args"] = {"resource": args}

        return normalized


class KubemedicObservation(Observation):
    """Structured Kubernetes cluster observation returned by KubeMedic."""

    t: int = 0
    scenario: str | None = None
    pods: list[PodObservation] = Field(default_factory=list)
    nodes: list[NodeObservation] = Field(default_factory=list)
