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
        observation = KubemedicObservation.model_validate(payload)

        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict) -> State:
        return State.model_validate(payload)
