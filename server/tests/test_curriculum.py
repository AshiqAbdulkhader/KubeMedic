"""Tests for curriculum-driven scenario progression."""

from __future__ import annotations

import random

from Kubemedic.server.curriculum import CurriculumController


def test_curriculum_prefers_untried_fault_types_before_repeats() -> None:
    curriculum = CurriculumController(
        rng=random.Random(0),
        fault_catalog={
            "fault_a": {"tier": 1, "min_difficulty": 0.0, "scenarios": ["KUBE-03"]},
            "fault_b": {"tier": 1, "min_difficulty": 0.0, "scenarios": ["KUBE-01"]},
        },
    )

    first_fault = curriculum.resolve_fault_type(curriculum.pick_scenario() or "")
    assert first_fault in {"fault_a", "fault_b"}

    curriculum.record(first_fault or "fault_a", True, 2, 20.0)
    second_fault = curriculum.resolve_fault_type(curriculum.pick_scenario() or "")

    assert second_fault in {"fault_a", "fault_b"}
    assert second_fault != first_fault


def test_curriculum_graduates_mastered_faults_and_fast_tracks_tier() -> None:
    curriculum = CurriculumController(
        fault_catalog={
            "oom_kill": {"tier": 1, "min_difficulty": 0.0, "scenarios": ["KUBE-03"]},
        }
    )

    for _ in range(3):
        curriculum.record("oom_kill", True, 2, 25.0)

    assert curriculum.get_tier_name() == "beginner"
    assert curriculum.get_graduated() == {"oom_kill"}
    assert curriculum.get_skill_profile()["oom_kill"] == 1.0


def test_curriculum_scales_judge_persona_with_difficulty() -> None:
    curriculum = CurriculumController(min_difficulty=0.75)

    assert curriculum.get_tier_name() == "advanced"
    assert curriculum.get_judge_persona() == "principal"
    assert "disk_pressure" in curriculum.get_unlocked_fault_types()
