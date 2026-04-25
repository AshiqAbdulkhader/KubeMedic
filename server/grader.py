"""Programmatic and LLM-based grading for KubeMedic episodes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .spec import SCENARIO_ROOT_CAUSES


DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_LLM_JUDGE_MODEL = "moonshotai/Kimi-K2.6:novita"
LLM_SCORE_FIELDS = (
    "diagnosis_quality",
    "k8s_knowledge",
    "tool_choice",
    "blast_radius",
    "root_cause_accuracy",
    "efficiency",
)
PATCH_TOOLS = {"kubectl_patch_resources", "kubectl_patch_tolerations"}


def programmatic_grade(episode_log: dict[str, Any]) -> dict[str, Any]:
    """Grade an episode using the deterministic rubric from the spec."""

    total_pods = int(episode_log["total_pods"])
    final_running = int(episode_log["final_running"])
    disruptions = int(episode_log["disruptions"])
    steps_taken = int(episode_log["steps_taken"])
    used_patch = bool(episode_log["used_patch_not_delete"])

    pod_restore_rate = (final_running / total_pods) if total_pods else 0.0
    pod_score = pod_restore_rate * 50
    disruption_score = 30 if disruptions == 0 else max(0, 30 - disruptions * 10)
    speed_score = max(0, 20 - steps_taken)
    tool_score = 10 if used_patch else 0
    total_score = pod_score + disruption_score + speed_score + tool_score

    return {
        "pod_restore_rate": pod_restore_rate,
        "disruptions": disruptions,
        "steps": steps_taken,
        "used_correct_tool": used_patch,
        "score": total_score,
        "max_score": 110,
    }


def _action_dict(action: Any) -> dict[str, Any]:
    if hasattr(action, "model_dump"):
        return action.model_dump()
    if isinstance(action, dict):
        return dict(action)
    return {
        "tool": getattr(action, "tool"),
        "args": dict(getattr(action, "args", {}) or {}),
    }


def _observation_dict(result_or_observation: Any) -> dict[str, Any]:
    observation = getattr(result_or_observation, "observation", result_or_observation)
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    if isinstance(observation, dict):
        return dict(observation)
    return {
        "t": getattr(observation, "t", 0),
        "scenario": getattr(observation, "scenario", None),
        "pods": list(getattr(observation, "pods", []) or []),
        "nodes": list(getattr(observation, "nodes", []) or []),
        "reward": getattr(observation, "reward", None),
        "done": getattr(observation, "done", False),
        "metadata": dict(getattr(observation, "metadata", {}) or {}),
    }


def build_transcript_step(
    *,
    step: int,
    action: Any,
    result: Any,
) -> dict[str, Any]:
    """Build a transcript entry from an action and step result."""

    action_payload = _action_dict(action)
    observation = _observation_dict(result)
    pods = observation.get("pods", []) or []
    info = (observation.get("metadata", {}) or {}).get("info", {}) or {}

    return {
        "step": step,
        "tool": action_payload.get("tool"),
        "args": action_payload.get("args", {}),
        "reward": float(observation.get("reward") or 0.0),
        "done": bool(observation.get("done", False)),
        "running": sum(1 for pod in pods if pod.get("phase") == "Running"),
        "disruptions": int(info.get("disruptions", 0) or 0),
        "blocked_reason": (observation.get("metadata", {}) or {}).get("blocked_reason"),
    }


def build_episode_log(
    *,
    final_result_or_observation: Any,
    transcript: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the deterministic grading input from the final observation and transcript."""

    observation = _observation_dict(final_result_or_observation)
    pods = observation.get("pods", []) or []
    metadata = observation.get("metadata", {}) or {}
    info = metadata.get("info", {}) or {}
    transcript_disruptions = [int(step.get("disruptions", 0) or 0) for step in transcript]

    return {
        "total_pods": len(pods),
        "final_running": sum(1 for pod in pods if pod.get("phase") == "Running"),
        "disruptions": int(
            info.get(
                "disruptions",
                max(transcript_disruptions) if transcript_disruptions else 0,
            )
            or 0
        ),
        "steps_taken": len(transcript),
        "used_patch_not_delete": any(
            step.get("tool") in PATCH_TOOLS for step in transcript
        ),
    }


def create_llm_judge_client(env_file: str | Path | None = None) -> OpenAI:
    """Create an OpenAI-compatible client pointed at the Hugging Face Router."""

    dotenv_path = Path(env_file) if env_file else DEFAULT_ENV_FILE
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required to use the LLM grader")

    return OpenAI(
        base_url=HF_ROUTER_BASE_URL,
        api_key=token,
    )


def _transcript_text(transcript: Any) -> str:
    if isinstance(transcript, str):
        return transcript
    return json.dumps(transcript, indent=2, sort_keys=True)


def _llm_judge_prompt(
    *,
    scenario_name: str,
    true_root_cause: str,
    transcript: Any,
    judge_persona: str,
) -> str:
    persona_guidance = {
        "junior": (
            "Judge persona: junior SRE mentor.\n"
            "Be lenient on efficiency, give credit for partial-but-correct diagnosis, "
            "and only score harshly for clearly risky actions.\n"
        ),
        "senior": (
            "Judge persona: senior SRE.\n"
            "Use balanced production standards for diagnosis quality, safety, and efficiency.\n"
        ),
        "principal": (
            "Judge persona: principal SRE reviewer.\n"
            "Be strict about unnecessary steps, weak diagnosis, and any avoidable blast radius. "
            "Reserve top scores for precise, efficient, production-grade incident handling.\n"
        ),
    }.get(judge_persona, "")

    return (
        "You are evaluating a Kubernetes SRE agent's incident response.\n"
        f"{persona_guidance}"
        "Score each dimension 0-10:\n\n"
        "1. diagnosis_quality   — Did it read kubectl_describe/logs before acting?\n"
        "                         Or did it blindly run commands?\n"
        "2. k8s_knowledge       — Does it understand QoS classes, eviction order,\n"
        "                         difference between requests and limits?\n"
        "3. tool_choice         — Did it use the right tool?\n"
        "                         (patch > delete, cordon before drain)\n"
        "4. blast_radius        — Did it avoid disrupting healthy Running pods?\n"
        "5. root_cause_accuracy — Did it correctly identify WHY the pod was broken?\n"
        "                         (not just that it was broken)\n"
        "6. efficiency          — Did it reach the fix in minimal steps, or scatter-query?\n\n"
        f"Scenario: {scenario_name}\n"
        f"True root cause: {true_root_cause}\n"
        "Full agent transcript:\n"
        f"{_transcript_text(transcript)}\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "diagnosis_quality": N,\n'
        '  "k8s_knowledge": N,\n'
        '  "tool_choice": N,\n'
        '  "blast_radius": N,\n'
        '  "root_cause_accuracy": N,\n'
        '  "efficiency": N,\n'
        '  "summary": "one sentence describing agent\'s key strength or weakness"\n'
        "}"
    )


def _extract_completion_text(completion: Any) -> str:
    message = completion.choices[0].message
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    text_parts.append(str(item["text"]))
            else:
                item_type = getattr(item, "type", None)
                if item_type == "text" and getattr(item, "text", None):
                    text_parts.append(str(item.text))
        return "\n".join(text_parts)

    return str(content)


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"LLM grader did not return a JSON object: {raw_text!r}")

    return json.loads(text[start : end + 1])


def _normalize_llm_scores(raw_scores: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field in LLM_SCORE_FIELDS:
        if field not in raw_scores:
            raise ValueError(f"LLM grader response is missing '{field}'")
        value = int(raw_scores[field])
        if not 0 <= value <= 10:
            raise ValueError(f"LLM grader field '{field}' must be between 0 and 10")
        normalized[field] = value

    summary = raw_scores.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("LLM grader response is missing a non-empty 'summary'")
    normalized["summary"] = summary.strip()
    return normalized


def llm_judge_grade(
    *,
    scenario_name: str,
    true_root_cause: str,
    transcript: Any,
    judge_persona: str = "senior",
    client: OpenAI | None = None,
    model: str = DEFAULT_LLM_JUDGE_MODEL,
) -> dict[str, Any]:
    """Call the Hugging Face Router LLM judge and parse its JSON response."""

    judge_client = client or create_llm_judge_client()
    completion = judge_client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Kubernetes incident grader. "
                    "Adapt your strictness to the requested judge persona. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": _llm_judge_prompt(
                    scenario_name=scenario_name,
                    true_root_cause=true_root_cause,
                    transcript=transcript,
                    judge_persona=judge_persona,
                ),
            },
        ],
    )
    raw_text = _extract_completion_text(completion)
    raw_scores = _extract_json_object(raw_text)
    scores = _normalize_llm_scores(raw_scores)
    scores["raw_response"] = raw_text
    return scores


def combined_final_score(
    *,
    programmatic: dict[str, Any],
    llm_scores: dict[str, Any],
) -> float:
    """Combine deterministic and LLM scores using the weighted formula from the spec."""

    llm_mean = sum(float(llm_scores[field]) for field in LLM_SCORE_FIELDS) / len(
        LLM_SCORE_FIELDS
    )
    return 0.6 * float(programmatic["score"]) + 0.4 * (llm_mean * 11)


def grade_episode(
    *,
    episode_log: dict[str, Any],
    scenario_name: str,
    true_root_cause: str,
    transcript: Any,
    judge_persona: str = "senior",
    client: OpenAI | None = None,
    model: str = DEFAULT_LLM_JUDGE_MODEL,
) -> dict[str, Any]:
    """Run both grading layers and return a combined result bundle."""

    programmatic = programmatic_grade(episode_log)
    llm_scores = llm_judge_grade(
        scenario_name=scenario_name,
        true_root_cause=true_root_cause,
        transcript=transcript,
        judge_persona=judge_persona,
        client=client,
        model=model,
    )
    final_score = combined_final_score(
        programmatic=programmatic,
        llm_scores=llm_scores,
    )
    return {
        "programmatic": programmatic,
        "llm": llm_scores,
        "final_score": final_score,
    }


def grade_recorded_episode(
    *,
    final_result_or_observation: Any,
    transcript: list[dict[str, Any]],
    scenario_name: str | None = None,
    true_root_cause: str | None = None,
    judge_persona: str = "senior",
    client: OpenAI | None = None,
    model: str = DEFAULT_LLM_JUDGE_MODEL,
) -> dict[str, Any]:
    """Grade a completed run directly from its final result and transcript."""

    observation = _observation_dict(final_result_or_observation)
    metadata = observation.get("metadata", {}) or {}
    resolved_scenario = scenario_name or observation.get("scenario")
    if not resolved_scenario:
        raise ValueError("scenario_name is required when the final observation has no scenario")

    resolved_root_cause = (
        true_root_cause
        or metadata.get("scenario_root_cause")
        or SCENARIO_ROOT_CAUSES.get(str(resolved_scenario))
    )
    if not resolved_root_cause:
        raise ValueError(
            "true_root_cause is required when it cannot be inferred from the scenario"
        )

    episode_log = build_episode_log(
        final_result_or_observation=observation,
        transcript=transcript,
    )
    graded = grade_episode(
        episode_log=episode_log,
        scenario_name=str(resolved_scenario),
        true_root_cause=str(resolved_root_cause),
        transcript=transcript,
        judge_persona=judge_persona,
        client=client,
        model=model,
    )
    graded["episode_log"] = episode_log
    graded["transcript"] = transcript
    return graded
