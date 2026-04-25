"""Curriculum controller for progressive KubeMedic training."""

from __future__ import annotations

import logging
import os
import random
from collections import defaultdict
from typing import Any


logger = logging.getLogger(__name__)

# Fault types are an orchestration layer over the concrete KUBE-* scenarios
# already implemented by the environment. Each episode still injects one fault.
DEFAULT_FAULT_CATALOG: dict[str, dict[str, Any]] = {
    "oom_kill": {
        "tier": 1,
        "min_difficulty": 0.0,
        "scenarios": ["KUBE-03"],
    },
    "memory_pressure": {
        "tier": 1,
        "min_difficulty": 0.0,
        "scenarios": ["KUBE-01"],
    },
    "unschedulable": {
        "tier": 2,
        "min_difficulty": 0.3,
        "scenarios": ["KUBE-04"],
    },
    "taint_mismatch": {
        "tier": 2,
        "min_difficulty": 0.3,
        "scenarios": ["KUBE-05"],
    },
    "disk_pressure": {
        "tier": 3,
        "min_difficulty": 0.6,
        "scenarios": ["KUBE-06"],
    },
}

MASTERY_THRESHOLD = 0.7
MASTERY_WINDOW = 10
MIN_EPISODES_FOR_MASTERY = 3

DIFFICULTY_TIERS = [
    {"name": "warmup", "max_diff": 0.25, "min_episodes": 5, "advance_rate": 0.6},
    {"name": "beginner", "max_diff": 0.40, "min_episodes": 5, "advance_rate": 0.6},
    {"name": "intermediate", "max_diff": 0.60, "min_episodes": 8, "advance_rate": 0.65},
    {"name": "advanced", "max_diff": 0.80, "min_episodes": 10, "advance_rate": 0.7},
    {"name": "expert", "max_diff": 0.95, "min_episodes": 0, "advance_rate": 1.0},
]


class CurriculumController:
    """Tracks mastery and picks progressively harder KubeMedic scenarios."""

    def __init__(
        self,
        *,
        fault_catalog: dict[str, dict[str, Any]] | None = None,
        rng: random.Random | None = None,
        min_difficulty: float | None = None,
        quality_threshold: float | None = None,
    ) -> None:
        self._fault_catalog = fault_catalog or DEFAULT_FAULT_CATALOG
        self._rng = rng or random.Random()

        self.history: dict[str, list[bool]] = defaultdict(list)
        self.step_counts: dict[str, list[int]] = defaultdict(list)
        self.episode_rewards: list[float] = []
        self.episode_history: list[dict[str, Any]] = []
        self.episode_count = 0
        self._tier_index = 0
        self._tier_episodes = 0
        self._graduated: set[str] = set()

        forced_min_difficulty = (
            min_difficulty
            if min_difficulty is not None
            else float(os.environ.get("EVAL_MIN_DIFFICULTY", "0.0"))
        )
        self._min_difficulty = max(0.0, forced_min_difficulty)
        self._quality_threshold = (
            quality_threshold
            if quality_threshold is not None
            else float(os.environ.get("CURRICULUM_QUALITY_THRESHOLD", "75.0"))
        )

        if self._min_difficulty > 0:
            for i, tier in enumerate(DIFFICULTY_TIERS):
                if tier["max_diff"] >= self._min_difficulty:
                    self._tier_index = i
                    break
            logger.info(
                "Curriculum: forced min_difficulty=%s, starting at tier=%s",
                self._min_difficulty,
                self.get_tier_name(),
            )

    def resolve_fault_type(self, scenario: str) -> str | None:
        """Return the curriculum fault type for a concrete scenario name."""
        normalized = scenario.upper()
        for fault_type, meta in self._fault_catalog.items():
            scenarios = [name.upper() for name in meta.get("scenarios", [])]
            if normalized in scenarios:
                return fault_type
        return None

    def record(
        self,
        fault_type: str,
        success: bool,
        steps: int,
        reward: float,
        *,
        quality_score: float | None = None,
    ) -> None:
        """Record an episode outcome and update progression state."""
        quality_pass = quality_score is None or quality_score >= self._quality_threshold
        mastery_success = success and quality_pass

        self.history[fault_type].append(mastery_success)
        self.step_counts[fault_type].append(steps)
        self.episode_rewards.append(reward)
        self.episode_history.append(
            {
                "fault_type": fault_type,
                "success": mastery_success,
                "solved": success,
                "quality_score": quality_score,
            }
        )
        self.episode_count += 1
        self._tier_episodes += 1
        self._maybe_advance_tier()

        recent = self.history[fault_type][-MASTERY_WINDOW:]
        if (
            len(recent) >= MIN_EPISODES_FOR_MASTERY
            and sum(recent) / len(recent) >= MASTERY_THRESHOLD
            and fault_type not in self._graduated
        ):
            self._graduated.add(fault_type)
            logger.info(
                "Curriculum: agent MASTERED '%s' (%s/%s successes) -- graduating",
                fault_type,
                sum(recent),
                len(recent),
            )

    def _maybe_advance_tier(self) -> None:
        if self._tier_index >= len(DIFFICULTY_TIERS) - 1:
            return

        tier = DIFFICULTY_TIERS[self._tier_index]
        recent_rate = self._recent_success_rate()
        fast_track = self._tier_episodes >= 3 and recent_rate >= 0.9

        if not fast_track and self._tier_episodes < tier["min_episodes"]:
            return

        if recent_rate >= tier["advance_rate"]:
            logger.info(
                "Curriculum: advancing from %s (rate=%.0f%%, episodes=%s%s)",
                tier["name"],
                recent_rate * 100,
                self._tier_episodes,
                ", FAST-TRACK" if fast_track else "",
            )
            self._tier_index += 1
            self._tier_episodes = 0

    def _recent_success_rate(self, window: int = 10) -> float:
        recent = self.episode_history[-window:]
        if not recent:
            return 0.0
        return sum(1 for episode in recent if episode["success"]) / len(recent)

    def get_skill_profile(self) -> dict[str, float]:
        return {
            fault_type: round(sum(results[-MASTERY_WINDOW:]) / len(results[-MASTERY_WINDOW:]), 2)
            for fault_type, results in self.history.items()
            if results
        }

    def get_weak_spots(self) -> list[str]:
        profile = self.get_skill_profile()
        return [fault_type for fault_type, rate in profile.items() if rate < MASTERY_THRESHOLD]

    def get_graduated(self) -> set[str]:
        return set(self._graduated)

    def get_unlocked_fault_types(self) -> list[str]:
        difficulty = self.get_difficulty()
        return [
            fault_type
            for fault_type, meta in self._fault_catalog.items()
            if float(meta["min_difficulty"]) <= difficulty
        ]

    def get_difficulty(self) -> float:
        tier = DIFFICULTY_TIERS[self._tier_index]
        if self.episode_count < 3 and self._min_difficulty == 0:
            return 0.15

        rate = self._recent_success_rate()
        tier_floor = 0.1 if self._tier_index == 0 else DIFFICULTY_TIERS[self._tier_index - 1]["max_diff"]
        natural = min(
            tier["max_diff"],
            tier_floor + rate * (tier["max_diff"] - tier_floor),
        )
        return max(natural, self._min_difficulty)

    def get_tier_name(self) -> str:
        return DIFFICULTY_TIERS[self._tier_index]["name"]

    def get_judge_persona(self) -> str:
        difficulty = self.get_difficulty()
        if difficulty < 0.4:
            return "junior"
        if difficulty < 0.7:
            return "senior"
        return "principal"

    def should_use_adversarial(self) -> bool:
        return "adversarial" in self._fault_catalog and self.get_difficulty() >= 0.6 and len(self._graduated) >= 2

    def pick_fault_type(self) -> str | None:
        """Pick the next fault type, or None when adversarial generation should take over."""
        if self.should_use_adversarial():
            return None

        unlocked = self.get_unlocked_fault_types()
        weak_spots = self.get_weak_spots()
        tried = set(self.history.keys())
        untried = [fault_type for fault_type in unlocked if fault_type not in tried and fault_type != "adversarial"]
        counts = {fault_type: len(results) for fault_type, results in self.history.items()}

        def over_limit(fault_type: str) -> bool:
            tier = int(self._fault_catalog.get(fault_type, {}).get("tier", 1))
            limit = 2 if tier == 1 else 3
            return counts.get(fault_type, 0) >= limit and bool(untried)

        if untried:
            return self._rng.choice(untried)

        weak_and_unlocked = [
            fault_type
            for fault_type in weak_spots
            if fault_type in unlocked and fault_type != "adversarial" and not over_limit(fault_type)
        ]
        if weak_and_unlocked:
            return self._rng.choice(weak_and_unlocked)

        candidates = [
            fault_type
            for fault_type in unlocked
            if fault_type != "adversarial" and not over_limit(fault_type)
        ]
        if not candidates:
            candidates = [fault_type for fault_type in unlocked if fault_type != "adversarial"]
        if not candidates:
            return None

        weights = [1 if fault_type in self._graduated else 3 for fault_type in candidates]
        return self._rng.choices(candidates, weights=weights, k=1)[0]

    def pick_scenario(self) -> str | None:
        """Pick the next concrete KUBE-* scenario for the current curriculum state."""
        fault_type = self.pick_fault_type()
        if fault_type is None:
            return None

        scenarios = list(self._fault_catalog.get(fault_type, {}).get("scenarios", []))
        if not scenarios:
            raise ValueError(f"No scenarios configured for fault type '{fault_type}'")
        return self._rng.choice(scenarios)

    def get_stats(self) -> dict[str, Any]:
        return {
            "episode_count": self.episode_count,
            "tier": self.get_tier_name(),
            "tier_episodes": self._tier_episodes,
            "difficulty": round(self.get_difficulty(), 2),
            "judge_persona": self.get_judge_persona(),
            "quality_threshold": self._quality_threshold,
            "skill_profile": self.get_skill_profile(),
            "graduated": sorted(self._graduated),
            "weak_spots": self.get_weak_spots(),
            "unlocked_faults": self.get_unlocked_fault_types(),
            "use_adversarial": self.should_use_adversarial(),
            "avg_reward_last_10": round(
                sum(self.episode_rewards[-10:]) / max(1, len(self.episode_rewards[-10:])),
                3,
            ),
        }
