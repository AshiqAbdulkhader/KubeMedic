"""Tests for the KubeMedic agent runner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Kubemedic.server.agent import run_episode_with_env


class FakeEnv:
    def __init__(self) -> None:
        self.steps = 0

    async def reset(self, scenario: str = "KUBE-03") -> SimpleNamespace:
        return SimpleNamespace(
            observation={
                "t": 0,
                "scenario": scenario,
                "pods": [{"name": "payment-svc-abc", "phase": "Failed"}],
                "nodes": [],
                "reward": 0.0,
                "done": False,
                "metadata": {
                    "scenario_root_cause": "OOMKilled due to low memory limit.",
                    "info": {"disruptions": 0, "steps_taken": 0},
                },
            },
            reward=0.0,
            done=False,
        )

    async def step(self, action) -> SimpleNamespace:
        self.steps += 1
        if self.steps == 1:
            return SimpleNamespace(
                observation={
                    "t": 1,
                    "scenario": "KUBE-03",
                    "pods": [{"name": "payment-svc-abc", "phase": "Failed"}],
                    "nodes": [],
                    "reward": 0.0,
                    "done": False,
                    "metadata": {
                        "scenario_root_cause": "OOMKilled due to low memory limit.",
                        "tool_result": {"name": "payment-svc-abc"},
                        "info": {"disruptions": 0, "steps_taken": 1},
                    },
                },
                reward=0.0,
                done=False,
            )

        return SimpleNamespace(
            observation={
                "t": 2,
                "scenario": "KUBE-03",
                "pods": [{"name": "payment-svc-abc", "phase": "Running"}],
                "nodes": [],
                "reward": 10.0,
                "done": True,
                "metadata": {
                    "scenario_root_cause": "OOMKilled due to low memory limit.",
                    "tool_result": {"deployment": "payment-svc"},
                    "info": {"disruptions": 0, "steps_taken": 2},
                },
            },
            reward=10.0,
            done=True,
        )


@pytest.fixture
def fake_model_client() -> SimpleNamespace:
    responses = iter(
        [
            '{"tool": "kubectl_describe", "args": {"resource": "pod", "name": "payment-svc-abc", "namespace": "challenge"}}',
            '{"tool": "kubectl_patch_resources", "args": {"deployment_name": "payment-svc", "namespace": "challenge", "container_name": "payment-svc", "limits_memory_mi": 512}}',
        ]
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
                )
            )
        )
    )


@pytest.fixture
def fake_grader_client() -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='{"diagnosis_quality": 9, "k8s_knowledge": 9, "tool_choice": 9, "blast_radius": 10, "root_cause_accuracy": 9, "efficiency": 8, "summary": "Diagnosed before patching the workload."}'
                            )
                        )
                    ]
                )
            )
        )
    )


@pytest.mark.anyio
async def test_run_episode_with_env_records_transcript_and_grading(
    fake_model_client: SimpleNamespace,
    fake_grader_client: SimpleNamespace,
) -> None:
    result = await run_episode_with_env(
        FakeEnv(),
        scenario="KUBE-03",
        client=fake_model_client,
        grader_client=fake_grader_client,
    )

    assert result["scenario"] == "KUBE-03"
    assert result["steps"] == 2
    assert result["transcript"][0]["tool"] == "kubectl_describe"
    assert result["transcript"][1]["tool"] == "kubectl_patch_resources"
    assert result["grading"]["programmatic"]["used_correct_tool"] is True
    assert result["grading"]["llm"]["blast_radius"] == 10


@pytest.mark.anyio
async def test_run_episode_with_env_records_invalid_json_turn(
    fake_grader_client: SimpleNamespace,
) -> None:
    responses = iter(
        [
            "not json",
            '{"tool": "kubectl_patch_resources", "args": {"deployment_name": "payment-svc", "namespace": "challenge", "container_name": "payment-svc", "limits_memory_mi": 512}}',
            '{"tool": "kubectl_patch_resources", "args": {"deployment_name": "payment-svc", "namespace": "challenge", "container_name": "payment-svc", "limits_memory_mi": 512}}',
        ]
    )
    bad_then_good_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
                )
            )
        )
    )

    result = await run_episode_with_env(
        FakeEnv(),
        scenario="KUBE-03",
        client=bad_then_good_client,
        grader_client=fake_grader_client,
        max_steps=3,
    )

    assert result["transcript"][0]["tool"] == "__invalid_json__"
    assert "error" in result["transcript"][0]
    assert result["steps"] == 3
