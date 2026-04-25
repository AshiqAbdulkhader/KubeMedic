"""LLM agent runner for KubeMedic episodes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from ..client import KubemedicEnv
from ..models import KubemedicAction
from .curriculum import CurriculumController
from .grader import (
    DEFAULT_ENV_FILE,
    DEFAULT_LLM_JUDGE_MODEL,
    HF_ROUTER_BASE_URL,
    build_episode_log,
    build_transcript_step,
    grade_recorded_episode,
)


DEFAULT_AGENT_MODEL = "moonshotai/Kimi-K2.6:novita"
SYSTEM_PROMPT = """You are an expert Kubernetes SRE on-call. A cluster has pods in broken states.
Your job: diagnose the root cause and fix all pods to Running state without
disrupting any currently healthy Running pods.

AVAILABLE TOOLS (call one per step, respond with JSON only):

kubectl_get(resource, namespace)
  resource: pods | nodes | deployments | events | pvc | pv
  -> Returns list of resources with current state

kubectl_describe(resource, name, namespace)
  -> Returns full details: conditions, events, container states, resource usage
  -> ALWAYS call this on broken pods before taking any action
  -> The Events section tells you exactly why a pod is failing

kubectl_logs(pod_name, namespace, previous=False)
  -> Returns container logs. Use previous=True to see why it crashed last time

kubectl_top_pods(namespace)
  -> Returns ACTUAL CPU/memory usage (vs requests/limits)
  -> Essential: if a pod requests 6Gi but uses 50Mi, the fix is to lower requests

kubectl_top_nodes()
  -> Returns actual node resource usage
  -> Helps identify which node is under pressure

kubectl_patch_resources(deployment_name, namespace, container_name, requests_memory_mi, limits_memory_mi, requests_cpu_m, limits_cpu_m)
  -> Fix OOMKill: increase limits_memory_mi above actual usage
  -> Fix Unschedulable: decrease requests_memory_mi to actual usage
  -> PREFERRED over delete - fixes the root cause, not just the symptom

kubectl_patch_tolerations(deployment_name, namespace, tolerations)
  -> Fix taint/toleration mismatch
  -> tolerations: [{"key": "X", "operator": "Equal", "value": "Y", "effect": "NoSchedule"}]

kubectl_cordon(node_name)
  -> Marks node unschedulable - safe, reversible

kubectl_uncordon(node_name)
  -> Re-enables scheduling on a node

kubectl_delete_pod(pod_name, namespace, force=False)
  -> ONLY use for pods in Evicted or Failed state that won't self-clean
  -> NEVER use on Running pods or as a fix for CrashLoopBackOff

kubectl_delete_workload(resource, name, namespace)
  -> Delete a workload object in the challenge namespace
  -> Use this when a broken controller itself is causing cluster pressure
  -> Supported resources: daemonset, deployment

STRATEGY - follow this order:
1. kubectl_get(pods) -> see the full picture
2. Treat a pod as broken if phase != Running, OR reason is non-empty, OR restarts > 0
3. kubectl_describe the broken pods -> read Events carefully
4. kubectl_top_pods -> check actual vs requested resources
5. kubectl_logs with previous=True -> for CrashLoopBackOff pods
6. Form a hypothesis about root cause BEFORE acting
7. Choose the most targeted fix (patch resources > adjust config > delete)
8. After fix, kubectl_get again to confirm pods are Running

KEY KUBERNETES FACTS:
- A pod can have phase=Running and still be broken if reason=CrashLoopBackOff or containers are not ready
- OOMKilled means memory limit is too LOW - increase limits, not requests
- Unschedulable "Insufficient memory" means requests are too HIGH - decrease requests
- QoS classes: Guaranteed (req==limits) > Burstable (req<limits) > BestEffort (none)
- BestEffort pods are evicted first under node pressure - this is expected behaviour
- CrashLoopBackOff + OOMKilled = always fix with kubectl_patch_resources, not delete
- DiskPressure caused by a log-flood DaemonSet should be fixed by deleting the DaemonSet and then cleaning up any stuck evicted pods

Respond ONLY with valid JSON: {"tool": "tool_name", "args": {"key": "value"}}"""


def create_agent_client(env_file: str | Path | None = None) -> OpenAI:
    """Create the HF Router OpenAI-compatible client for agent inference."""

    dotenv_path = Path(env_file) if env_file else DEFAULT_ENV_FILE
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required to run the KubeMedic agent")

    return OpenAI(
        base_url=HF_ROUTER_BASE_URL,
        api_key=hf_token,
    )


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
        raise ValueError(f"Model response did not contain a JSON object: {raw_text!r}")
    return json.loads(text[start : end + 1])


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
                if getattr(item, "type", None) == "text" and getattr(item, "text", None):
                    text_parts.append(str(item.text))
        return "\n".join(text_parts)

    return str(content)


def _observation_payload(result_or_observation: Any) -> dict[str, Any]:
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


async def run_episode_with_env(
    env: Any,
    *,
    scenario: str | None = "KUBE-03",
    max_steps: int = 30,
    client: OpenAI | None = None,
    model: str = DEFAULT_AGENT_MODEL,
    grader_client: OpenAI | None = None,
    grader_model: str = DEFAULT_LLM_JUDGE_MODEL,
    grade: bool = True,
    curriculum: CurriculumController | None = None,
    judge_persona: str | None = None,
) -> dict[str, Any]:
    """Run an agent episode against an already-constructed async env client."""

    model_client = client or create_agent_client()
    selected_scenario = scenario
    selected_fault_type = None
    curriculum_state_before = None

    if curriculum is not None:
        curriculum_state_before = curriculum.get_stats()
        if selected_scenario is None:
            selected_scenario = curriculum.pick_scenario()
        if selected_scenario is None:
            raise RuntimeError(
                "Curriculum requested adversarial scenario generation, "
                "but no adversarial scenario generator is configured"
            )
        selected_fault_type = curriculum.resolve_fault_type(selected_scenario)
        judge_persona = judge_persona or curriculum.get_judge_persona()

    selected_scenario = selected_scenario or "KUBE-03"
    judge_persona = judge_persona or "senior"

    reset_result = await env.reset(scenario=selected_scenario)
    observation = _observation_payload(reset_result)
    history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    transcript: list[dict[str, Any]] = []
    total_reward = float(observation.get("reward") or 0.0)
    final_result: Any = reset_result

    for step_index in range(1, max_steps + 1):
        user_message = (
            f"Step {step_index}/{max_steps}\n\nCurrent cluster state:\n"
            f"{json.dumps(observation, indent=2, sort_keys=True)}"
        )
        history.append({"role": "user", "content": user_message})

        completion = model_client.chat.completions.create(
            model=model,
            temperature=0,
            messages=history,
        )
        raw_text = _extract_completion_text(completion).strip()
        history.append({"role": "assistant", "content": raw_text})

        try:
            action_dict = _extract_json_object(raw_text)
            action = KubemedicAction(
                tool=action_dict["tool"],
                args=action_dict.get("args", {}),
            )
        except Exception as exc:
            info = (observation.get("metadata", {}) or {}).get("info", {}) or {}
            transcript.append(
                {
                    "step": step_index,
                    "tool": "__invalid_json__",
                    "args": {},
                    "reward": 0.0,
                    "done": False,
                    "running": sum(
                        1 for pod in (observation.get("pods", []) or []) if pod.get("phase") == "Running"
                    ),
                    "disruptions": int(info.get("disruptions", 0) or 0),
                    "blocked_reason": None,
                    "model_response": raw_text,
                    "error": str(exc),
                }
            )
            continue

        result = await env.step(action)
        final_result = result
        observation = _observation_payload(result)
        reward = float(observation.get("reward") or 0.0)
        total_reward += reward

        transcript_step = build_transcript_step(
            step=step_index,
            action=action,
            result=result,
        )
        transcript_step["model_response"] = raw_text
        transcript.append(transcript_step)

        if observation.get("done", False):
            break

    grading = None
    if grade:
        grading = grade_recorded_episode(
            final_result_or_observation=final_result,
            transcript=transcript,
            judge_persona=judge_persona,
            client=grader_client or model_client,
            model=grader_model,
        )

    final_observation = _observation_payload(final_result)
    episode_log = (
        grading["episode_log"]
        if grading is not None
        else build_episode_log(
            final_result_or_observation=final_result,
            transcript=transcript,
        )
    )
    solved = bool(final_observation.get("done", False)) and (
        int(episode_log["final_running"]) == int(episode_log["total_pods"]) and int(episode_log["total_pods"]) > 0
    )

    if curriculum is not None:
        curriculum.record(
            selected_fault_type or selected_scenario.lower(),
            solved,
            len(transcript),
            total_reward,
        )

    return {
        "scenario": selected_scenario,
        "fault_type": selected_fault_type,
        "judge_persona": judge_persona,
        "model": model,
        "steps": len(transcript),
        "solved": solved,
        "total_reward": total_reward,
        "final_observation": final_observation,
        "transcript": transcript,
        "grading": grading,
        "curriculum": (
            {
                "before": curriculum_state_before,
                "after": curriculum.get_stats(),
            }
            if curriculum is not None
            else None
        ),
    }


async def run_episode(
    *,
    base_url: str,
    scenario: str | None = "KUBE-03",
    max_steps: int = 30,
    client: OpenAI | None = None,
    model: str = DEFAULT_AGENT_MODEL,
    grader_client: OpenAI | None = None,
    grader_model: str = DEFAULT_LLM_JUDGE_MODEL,
    grade: bool = True,
    curriculum: CurriculumController | None = None,
    judge_persona: str | None = None,
) -> dict[str, Any]:
    """Connect to an OpenEnv server and run one KubeMedic agent episode."""

    async with KubemedicEnv(base_url=base_url) as env:
        return await run_episode_with_env(
            env,
            scenario=scenario,
            max_steps=max_steps,
            client=client,
            model=model,
            grader_client=grader_client,
            grader_model=grader_model,
            grade=grade,
            curriculum=curriculum,
            judge_persona=judge_persona,
        )


async def run_curriculum(
    *,
    base_url: str,
    episodes: int,
    max_steps: int = 30,
    client: OpenAI | None = None,
    model: str = DEFAULT_AGENT_MODEL,
    grader_client: OpenAI | None = None,
    grader_model: str = DEFAULT_LLM_JUDGE_MODEL,
    grade: bool = True,
    curriculum: CurriculumController | None = None,
) -> dict[str, Any]:
    """Run multiple episodes while letting the curriculum pick each scenario."""

    controller = curriculum or CurriculumController()
    results: list[dict[str, Any]] = []

    async with KubemedicEnv(base_url=base_url) as env:
        for _ in range(episodes):
            results.append(
                await run_episode_with_env(
                    env,
                    scenario=None,
                    max_steps=max_steps,
                    client=client,
                    model=model,
                    grader_client=grader_client,
                    grader_model=grader_model,
                    grade=grade,
                    curriculum=controller,
                )
            )

    return {
        "episodes": episodes,
        "results": results,
        "curriculum": controller.get_stats(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--scenario", default="KUBE-03")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--use-curriculum", action="store_true")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--grader-model", default=DEFAULT_LLM_JUDGE_MODEL)
    parser.add_argument("--no-grade", action="store_true")
    args = parser.parse_args()

    if args.use_curriculum:
        result = asyncio.run(
            run_curriculum(
                base_url=args.base_url,
                episodes=args.episodes,
                max_steps=args.max_steps,
                model=args.model,
                grader_model=args.grader_model,
                grade=not args.no_grade,
            )
        )
    else:
        result = asyncio.run(
            run_episode(
                base_url=args.base_url,
                scenario=args.scenario,
                max_steps=args.max_steps,
                model=args.model,
                grader_model=args.grader_model,
                grade=not args.no_grade,
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
