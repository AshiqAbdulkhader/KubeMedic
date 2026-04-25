"""Tests for the KubeMedic grading helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from Kubemedic.server.grader import (
    build_episode_log,
    build_transcript_step,
    combined_final_score,
    create_llm_judge_client,
    grade_episode,
    grade_recorded_episode,
    llm_judge_grade,
    programmatic_grade,
)


def test_programmatic_grade_matches_spec_formula() -> None:
    result = programmatic_grade(
        {
            "total_pods": 5,
            "final_running": 4,
            "disruptions": 1,
            "steps_taken": 7,
            "used_patch_not_delete": True,
        }
    )

    assert result == {
        "pod_restore_rate": 0.8,
        "disruptions": 1,
        "steps": 7,
        "used_correct_tool": True,
        "score": 83.0,
        "max_score": 110,
    }


def test_combined_final_score_uses_weighted_formula() -> None:
    final = combined_final_score(
        programmatic={"score": 80},
        llm_scores={
            "diagnosis_quality": 8,
            "k8s_knowledge": 7,
            "tool_choice": 9,
            "blast_radius": 10,
            "root_cause_accuracy": 8,
            "efficiency": 6,
            "summary": "Good diagnosis.",
        },
    )

    assert final == pytest.approx(83.2)


def test_build_transcript_step_extracts_run_metadata() -> None:
    step = build_transcript_step(
        step=2,
        action={"tool": "kubectl_patch_resources", "args": {"deployment_name": "payment-svc"}},
        result={
            "reward": 10,
            "done": False,
            "pods": [
                {"name": "payment-svc-abc", "phase": "Running"},
                {"name": "api-gw-123", "phase": "Running"},
            ],
            "metadata": {"info": {"disruptions": 1}},
        },
    )

    assert step == {
        "step": 2,
        "tool": "kubectl_patch_resources",
        "args": {"deployment_name": "payment-svc"},
        "reward": 10.0,
        "done": False,
        "running": 2,
        "disruptions": 1,
        "blocked_reason": None,
    }


def test_build_episode_log_derives_grader_input_from_final_observation() -> None:
    transcript = [
        {"step": 1, "tool": "kubectl_describe", "disruptions": 0},
        {"step": 2, "tool": "kubectl_patch_resources", "disruptions": 1},
    ]
    episode_log = build_episode_log(
        final_result_or_observation={
            "scenario": "KUBE-03",
            "pods": [
                {"name": "payment-svc-abc", "phase": "Running"},
                {"name": "api-gw-123", "phase": "Running"},
                {"name": "order-svc-123", "phase": "Pending"},
            ],
            "metadata": {"info": {"disruptions": 1}},
        },
        transcript=transcript,
    )

    assert episode_log == {
        "total_pods": 3,
        "final_running": 2,
        "disruptions": 1,
        "steps_taken": 2,
        "used_patch_not_delete": True,
    }


def test_create_llm_judge_client_loads_hf_token_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('HF_TOKEN="test-token"\n', encoding="utf-8")

    with patch.dict("os.environ", {}, clear=True):
        client = create_llm_judge_client(env_file)

    assert client.base_url == "https://router.huggingface.co/v1/"
    assert client.api_key == "test-token"


def test_llm_judge_grade_parses_json_only_response() -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="""```json
{
  "diagnosis_quality": 8,
  "k8s_knowledge": 9,
  "tool_choice": 8,
  "blast_radius": 10,
  "root_cause_accuracy": 9,
  "efficiency": 7,
  "summary": "Used the right fix with minimal blast radius."
}
```"""
                )
            )
        ]
    )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: completion)
        )
    )

    result = llm_judge_grade(
        scenario_name="KUBE-03",
        true_root_cause="OOMKilled due to low memory limit.",
        transcript=[{"step": 1, "tool": "kubectl_describe"}],
        client=fake_client,
    )

    assert result["diagnosis_quality"] == 8
    assert result["blast_radius"] == 10
    assert result["summary"] == "Used the right fix with minimal blast radius."


def test_grade_episode_returns_programmatic_llm_and_final_score() -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"diagnosis_quality": 8, "k8s_knowledge": 8, "tool_choice": 8, "blast_radius": 9, "root_cause_accuracy": 8, "efficiency": 7, "summary": "Solid recovery path."}'
                )
            )
        ]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: completion)
        )
    )

    result = grade_episode(
        episode_log={
            "total_pods": 4,
            "final_running": 4,
            "disruptions": 0,
            "steps_taken": 9,
            "used_patch_not_delete": True,
        },
        scenario_name="KUBE-03",
        true_root_cause="OOMKilled due to low memory limit.",
        transcript=[{"step": 1, "tool": "kubectl_describe"}],
        client=fake_client,
    )

    assert result["programmatic"]["score"] == 101.0
    assert result["llm"]["tool_choice"] == 8
    assert result["final_score"] == pytest.approx(95.8)


def test_grade_recorded_episode_wires_episode_log_and_grader_together() -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"diagnosis_quality": 9, "k8s_knowledge": 8, "tool_choice": 9, "blast_radius": 9, "root_cause_accuracy": 9, "efficiency": 8, "summary": "Diagnosed cleanly and patched the root cause."}'
                )
            )
        ]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: completion)
        )
    )
    transcript = [
        build_transcript_step(
            step=1,
            action={"tool": "kubectl_describe", "args": {"resource": "pod", "name": "payment-svc-abc"}},
            result={
                "reward": 0,
                "done": False,
                "pods": [{"name": "payment-svc-abc", "phase": "Failed"}],
                "metadata": {"info": {"disruptions": 0}},
            },
        ),
        build_transcript_step(
            step=2,
            action={"tool": "kubectl_patch_resources", "args": {"deployment_name": "payment-svc"}},
            result={
                "reward": 10,
                "done": True,
                "scenario": "KUBE-03",
                "pods": [{"name": "payment-svc-abc", "phase": "Running"}],
                "metadata": {
                    "scenario_root_cause": "OOMKilled due to low memory limit.",
                    "info": {"disruptions": 0},
                },
            },
        ),
    ]

    result = grade_recorded_episode(
        final_result_or_observation={
            "scenario": "KUBE-03",
            "pods": [{"name": "payment-svc-abc", "phase": "Running"}],
            "metadata": {
                "scenario_root_cause": "OOMKilled due to low memory limit.",
                "info": {"disruptions": 0},
            },
        },
        transcript=transcript,
        client=fake_client,
    )

    assert result["episode_log"] == {
        "total_pods": 1,
        "final_running": 1,
        "disruptions": 0,
        "steps_taken": 2,
        "used_patch_not_delete": True,
    }
    assert result["programmatic"]["score"] == 108.0
    assert result["llm"]["root_cause_accuracy"] == 9
