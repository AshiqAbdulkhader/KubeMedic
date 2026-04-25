# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kubemedic Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import KubemedicAction, KubemedicObservation


class KubemedicEnv(
    EnvClient[KubemedicAction, KubemedicObservation, State]
):
    """Async client for the KubeMedic OpenEnv server."""

    def _step_payload(self, action: KubemedicAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[KubemedicObservation]:
        observation_payload = payload.get("observation", payload)
        if not isinstance(observation_payload, dict):
            observation_payload = {}

        merged_payload = dict(observation_payload)
        if "reward" not in merged_payload and "reward" in payload:
            merged_payload["reward"] = payload.get("reward")
        if "done" not in merged_payload and "done" in payload:
            merged_payload["done"] = payload.get("done")
        if "metadata" not in merged_payload and "metadata" in payload:
            merged_payload["metadata"] = payload.get("metadata")

        observation = KubemedicObservation.model_validate(merged_payload)

        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict) -> State:
        return State.model_validate(payload)
