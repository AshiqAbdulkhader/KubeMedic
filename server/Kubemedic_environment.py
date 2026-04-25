# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenEnv adapter for the KubeMedic environment implementation."""

from __future__ import annotations

import asyncio

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import KubemedicAction, KubemedicObservation
    from .env import KubeMedicEnv
except ImportError:
    from models import KubemedicAction, KubemedicObservation
    from server.env import KubeMedicEnv


class KubemedicEnvironment(Environment):
    """OpenEnv-compatible wrapper around the AKS-backed KubeMedic env."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        self.env = KubeMedicEnv()

    def reset(self, **kwargs: object) -> KubemedicObservation:
        return asyncio.run(self.reset_async(**kwargs))

    async def reset_async(self, **kwargs: object) -> KubemedicObservation:
        scenario = str(kwargs.get("scenario", "KUBE-03"))
        return await self.env.reset(scenario=scenario)

    def step(self, action: KubemedicAction, **kwargs: object) -> KubemedicObservation:  # type: ignore[override]
        return asyncio.run(self.step_async(action, **kwargs))

    async def step_async(
        self,
        action: KubemedicAction,
        **kwargs: object,
    ) -> KubemedicObservation:
        return await self.env.step(action)

    @property
    def state(self) -> State:
        return self.env.state

    def close(self) -> None:
        self.env.close()
