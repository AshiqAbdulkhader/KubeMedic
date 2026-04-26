"""Single-session GRPO training entrypoint for KubeMedic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kubemedic-mpl"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import transformers
from datasets import Dataset
from packaging.version import Version
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

from .client import KubemedicEnv
from .models import KubemedicAction, KubemedicObservation, ToolName

try:
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
except Exception:  # pragma: no cover - optional runtime dependency path
    ConnectionClosedError = Exception
    ConnectionClosedOK = Exception

SYSTEM_PROMPT = """You are a Kubernetes SRE agent operating KubeMedic.
Diagnose the cluster carefully, use the available actions, minimize blast radius, and stop once the cluster is healthy.

Respond with exactly one action per turn using this format:
- TOOL_NAME key=value key=value
- TOOL_NAME {"json": "object"}

Allowed tool names:
kubectl_get
kubectl_describe
kubectl_logs
kubectl_top_pods
kubectl_top_nodes
kubectl_patch_resources
kubectl_patch_tolerations
kubectl_cordon
kubectl_uncordon
kubectl_delete_pod
kubectl_delete_workload

Rules:
- Never include markdown fences or explanations.
- Prefer inspect-first behavior before mutating resources.
- Use namespace=challenge unless the observation clearly requires something else.
"""

TRAINING_HINT = (
    "Diagnose and repair this Kubernetes incident. Output one action only, in the required syntax."
)
DEFAULT_SCENARIOS = ["KUBE-01", "KUBE-03", "KUBE-04", "KUBE-05", "KUBE-06"]
VALID_TOOLS = set(ToolName.__args__)  # type: ignore[attr-defined]
CONNECT_TIMEOUT_S = 30.0
MESSAGE_TIMEOUT_S = 240.0
DEFAULT_ENV_FILES = [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-session GRPO KubeMedic agent.")
    parser.add_argument("--env-url", required=True, help="OpenEnv base URL for the KubeMedic server.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", default="outputs/kubemedic-qwen25-3b-grpo")
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--no-vllm", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--scenario-list", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--wandb-project", default="kubemedic-grpo")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_repo_env() -> None:
    for env_path in DEFAULT_ENV_FILES:
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def configure_wandb(project: str, run_name: str | None) -> list[str]:
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not api_key:
        os.environ["WANDB_DISABLED"] = "true"
        return []

    os.environ["WANDB_DISABLED"] = "false"
    os.environ.setdefault("WANDB_PROJECT", project)
    if run_name:
        os.environ.setdefault("WANDB_NAME", run_name)
    return ["wandb"]


def format_observation(obs: KubemedicObservation) -> str:
    pods = []
    for pod in obs.pods:
        parts = [f"phase={pod.phase}"]
        if pod.reason:
            parts.append(f"reason={pod.reason}")
        if pod.restarts:
            parts.append(f"restarts={pod.restarts}")
        pods.append(f"- {pod.name} ({pod.namespace}): " + " ".join(parts))

    nodes = []
    for node in obs.nodes:
        pressure = [c.type for c in node.conditions if c.status == "True" and c.type.endswith("Pressure")]
        nodes.append(
            f"- {node.name}: ready={node.ready} pressure={','.join(pressure) if pressure else 'none'}"
        )

    info = obs.info or {}
    reward_breakdown = info.get("reward_breakdown", {})
    breakdown_text = ", ".join(f"{k}={v:.2f}" for k, v in reward_breakdown.items()) if reward_breakdown else "none"

    return (
        f"{TRAINING_HINT}\n\n"
        f"Scenario: {obs.scenario}\n"
        f"Time step: {obs.t}\n"
        f"Blocked reason: {obs.blocked_reason or 'none'}\n"
        f"Last reward: {float(obs.reward or 0.0):.2f}\n"
        f"Disruptions: {info.get('disruptions', 0)}\n"
        f"Reward breakdown: {breakdown_text}\n\n"
        f"Pods:\n{chr(10).join(pods) if pods else '- none'}\n\n"
        f"Nodes:\n{chr(10).join(nodes) if nodes else '- none'}"
    )


def format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = ["Previous actions:"]
    for item in history:
        output = item["output"]
        if len(output) > 220:
            output = output[:220] + "..."
        lines.append(f"- {item['action']} -> reward={item['reward']:.2f} | {output}")
    return "\n".join(lines)


def coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none" or lowered == "null":
        return None
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return int(value)
    except ValueError:
        pass
    return value


def parse_action_text(text: str) -> KubemedicAction | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
    line = cleaned.splitlines()[0].strip()
    if not line:
        return None

    parts = line.split(maxsplit=1)
    tool = parts[0].strip()
    if tool not in VALID_TOOLS:
        return None

    if len(parts) == 1:
        return KubemedicAction(tool=tool, args={})

    remainder = parts[1].strip()
    if not remainder:
        return KubemedicAction(tool=tool, args={})

    args: dict[str, Any] = {}
    if remainder.startswith("{"):
        parsed = json.loads(remainder)
        if not isinstance(parsed, dict):
            return None
        args = parsed
    else:
        for token in shlex.split(remainder):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            args[key] = coerce_value(value)

    return KubemedicAction(tool=tool, args=args)


def build_dataset(num_rows: int) -> Dataset:
    rows = [{"prompt": TRAINING_HINT} for _ in range(num_rows)]
    return Dataset.from_list(rows)


def make_env(env_url: str):
    return KubemedicEnv(
        base_url=env_url,
        connect_timeout_s=CONNECT_TIMEOUT_S,
        message_timeout_s=MESSAGE_TIMEOUT_S,
    ).sync()


def render_prompt(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    system_lines = [msg["content"] for msg in messages if msg["role"] == "system"]
    user_lines = [msg["content"] for msg in messages if msg["role"] == "user"]
    return "\n\n".join(
        [
            *(["SYSTEM:\n" + "\n".join(system_lines)] if system_lines else []),
            *(["USER:\n" + "\n".join(user_lines)] if user_lines else []),
            "ASSISTANT:\n",
        ]
    )


def is_connection_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        isinstance(exc, (ConnectionClosedError, ConnectionClosedOK))
        or "ConnectionClosed" in exc.__class__.__name__
        or "no close frame received or sent" in text
        or "close frame" in text
        or "websocket" in text.lower()
    )


def smoke_test(env_url: str, scenario: str) -> None:
    env = make_env(env_url)
    try:
        result = env.reset(scenario=scenario)
        print(format_observation(result.observation)[:1200])
        step_result = env.step(KubemedicAction(tool="kubectl_get", args={"resource": "pods", "namespace": "challenge"}))
        print("\n--- smoke tool call ---\n")
        print(format_observation(step_result.observation)[:1200])
        print("\nSmoke test passed.")
    finally:
        env.close()


def generate_rollout_completions_local(
    trainer: GRPOTrainer,
    prompts: list[str],
) -> list[dict[str, Any]]:
    tokenizer = trainer.processing_class
    model = trainer.model
    results: list[dict[str, Any]] = []

    for prompt_text in prompts:
        encoded = tokenizer(prompt_text, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        generation = model.generate(
            **encoded,
            do_sample=True,
            temperature=trainer.temperature,
            top_p=trainer.top_p if trainer.top_p is not None else 1.0,
            top_k=trainer.top_k if trainer.top_k is not None else 50,
            max_new_tokens=trainer.max_completion_length,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        sequence = generation.sequences[0]
        prompt_len = encoded["input_ids"].shape[1]
        completion_ids = sequence[prompt_len:]
        if generation.scores:
            transition_scores = model.compute_transition_scores(
                generation.sequences,
                generation.scores,
                normalize_logits=True,
            )[0]
            logprobs = [float(score) for score in transition_scores.tolist()[: len(completion_ids)]]
        else:
            logprobs = []

        results.append(
            {
                "prompt_ids": sequence[:prompt_len].tolist(),
                "completion_ids": completion_ids.tolist(),
                "logprobs": logprobs,
                "text": tokenizer.decode(completion_ids, skip_special_tokens=True),
            }
        )

    return results


def rollout_once(
    trainer: GRPOTrainer,
    env_url: str,
    tokenizer: AutoTokenizer,
    scenario: str,
    max_turns: int,
    max_total_tokens: int,
    max_retries: int = 3,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        env = make_env(env_url)
        try:
            result = env.reset(scenario=scenario)
            observation = result.observation

            prompt_ids: list[int] = []
            completion_ids: list[int] = []
            logprobs: list[float] = []
            step_rewards: list[float] = []
            history: list[dict[str, Any]] = []

            for _ in range(max_turns):
                if result.done or len(completion_ids) >= max_total_tokens:
                    break

                user_parts = [format_observation(observation)]
                history_text = format_history(history)
                if history_text:
                    user_parts.append(history_text)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "\n\n".join(user_parts)},
                ]

                prompt_text = render_prompt(tokenizer, messages)
                if trainer.use_vllm:
                    rollout_output = generate_rollout_completions(trainer, [prompt_text])[0]
                else:
                    rollout_output = generate_rollout_completions_local(trainer, [prompt_text])[0]
                prompt_ids.extend(list(rollout_output["prompt_ids"]))
                completion_ids.extend(list(rollout_output["completion_ids"]))
                logprobs.extend([float(v) for v in rollout_output["logprobs"]])

                completion_text = (rollout_output.get("text") or "").strip()
                action = parse_action_text(completion_text)
                if action is None:
                    step_rewards.append(-1.0)
                    history.append(
                        {
                            "action": completion_text or "(empty)",
                            "output": "Invalid action format.",
                            "reward": -1.0,
                        }
                    )
                    continue

                result = env.step(action)
                observation = result.observation
                reward = float(result.reward or 0.0)
                step_rewards.append(reward)
                tool_output = json.dumps(observation.tool_result or {}, default=str)
                history.append({"action": completion_text, "output": tool_output, "reward": reward})

            total_reward = sum(step_rewards) if step_rewards else -1.0
            return {
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "logprobs": logprobs,
                "total_reward": total_reward,
                "steps": len(step_rewards),
                "scenario": scenario,
                "resolved": bool(result.done),
            }
        except Exception as exc:
            last_exc = exc
            if not is_connection_error(exc) or attempt >= max_retries:
                raise
            wait_s = min(2 ** (attempt - 1), 8)
            print(
                f"Episode websocket dropped during scenario {scenario} "
                f"(attempt {attempt}/{max_retries}): {exc}. Retrying in {wait_s}s..."
            )
            time.sleep(wait_s)
        finally:
            try:
                env.close()
            except Exception:
                pass

    raise RuntimeError(f"Episode failed after {max_retries} reconnect attempts: {last_exc}")


def reward_total(completions: list[str], **kwargs) -> list[float]:
    rewards = kwargs.get("total_reward") if kwargs else None
    return [float(value) for value in rewards] if rewards else [0.0 for _ in completions]


def plot_rewards(csv_path: Path, out_path: Path) -> None:
    rows = pd.read_csv(csv_path)
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rows["episode"], rows["total_reward"], marker="o", alpha=0.35, label="episode reward")
    rolling = rows["total_reward"].rolling(window=min(10, len(rows)), min_periods=1).mean()
    ax.plot(rows["episode"], rolling, linewidth=2, label="rolling mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("KubeMedic Single-Session GRPO Reward Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_trainer_metrics(log_history: pd.DataFrame, out_path: Path) -> None:
    if log_history.empty:
        return
    cols = [name for name in ["loss", "grad_norm", "learning_rate"] if name in log_history.columns]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        subset = log_history.dropna(subset=[col])
        if subset.empty or "step" not in subset.columns:
            continue
        sns.lineplot(data=subset, x="step", y=col, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def infer_lora_target_modules(model: AutoModelForCausalLM) -> list[str]:
    model_type = getattr(model.config, "model_type", "")
    if model_type in {"qwen2", "qwen2_5", "qwen3"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if model_type in {"gpt2"}:
        return ["c_attn", "c_proj", "c_fc"]

    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "c_attn", "c_proj", "c_fc"]
    inferred = [name for name in preferred if name in module_names]
    if inferred:
        return inferred
    raise ValueError(f"Could not infer LoRA target modules for model_type={model_type!r}.")


def main() -> None:
    args = parse_args()
    load_repo_env()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [item.strip() for item in args.scenario_list.split(",") if item.strip()]

    set_seed(args.seed)
    report_to = configure_wandb(args.wandb_project, args.wandb_run_name)
    if not args.skip_smoke_test:
        smoke_test(args.env_url, scenarios[0])
    if args.smoke_only:
        return
    if args.per_device_train_batch_size % args.num_generations != 0:
        raise ValueError("per_device_train_batch_size must be divisible by num_generations for GRPO.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_vllm = torch.cuda.is_available() and not args.no_vllm
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    quant_config = None
    if torch.cuda.is_available() and not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "quantization_config": quant_config,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        **model_kwargs,
    )

    if Version(transformers.__version__) < Version("4.56.0"):
        raise RuntimeError("transformers>=4.56.0 is required for the training entrypoint.")

    train_dataset = build_dataset(args.train_episodes)
    eval_dataset = build_dataset(args.eval_episodes)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=infer_lora_target_modules(model),
    )

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=1,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=report_to,
        use_vllm=use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        log_completions=False,
        max_tool_calling_iterations=1,
        seed=args.seed,
    )

    reward_log_path = output_dir / "episode_rewards.csv"
    transcript_path = output_dir / "agent_transcripts.jsonl"
    with reward_log_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "scenario", "total_reward", "steps", "resolved"])

    episode_counter = {"value": 0}

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        prompt_count = len(prompts)
        episode_prompt_ids: list[list[int]] = []
        episode_completion_ids: list[list[int]] = []
        episode_logprobs: list[list[float]] = []
        total_rewards: list[float] = []
        steps: list[int] = []
        scenario_names: list[str] = []
        resolved_flags: list[bool] = []

        for start in range(0, prompt_count, args.num_generations):
            scenario = random.choice(scenarios)
            group_size = min(args.num_generations, prompt_count - start)
            for _ in range(group_size):
                episode = rollout_once(
                    trainer=trainer,
                    env_url=args.env_url,
                    tokenizer=tokenizer,
                    scenario=scenario,
                    max_turns=args.max_turns,
                    max_total_tokens=4096,
                )
                episode_prompt_ids.append(episode["prompt_ids"])
                episode_completion_ids.append(episode["completion_ids"])
                episode_logprobs.append(episode["logprobs"])
                total_rewards.append(float(episode["total_reward"]))
                steps.append(int(episode["steps"]))
                scenario_names.append(episode["scenario"])
                resolved_flags.append(bool(episode["resolved"]))

                episode_counter["value"] += 1
                with reward_log_path.open("a", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            episode_counter["value"],
                            episode["scenario"],
                            episode["total_reward"],
                            episode["steps"],
                            episode["resolved"],
                        ]
                    )
                with transcript_path.open("a") as handle:
                    handle.write(json.dumps(episode) + "\n")

        return {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs": episode_logprobs,
            "total_reward": total_rewards,
            "steps": steps,
            "scenario": scenario_names,
            "resolved": resolved_flags,
        }

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_total,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=grpo_config,
        rollout_func=rollout_func,
        peft_config=peft_config,
    )

    trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    plot_rewards(reward_log_path, output_dir / "reward_plot.png")
    log_history = pd.DataFrame(trainer.state.log_history)
    log_history.to_csv(output_dir / "trainer_log_history.csv", index=False)
    plot_trainer_metrics(log_history, output_dir / "trainer_metrics.png")
    summary = {
        "model_id": args.model_id,
        "env_url": args.env_url,
        "train_episodes": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "num_generations": args.num_generations,
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
