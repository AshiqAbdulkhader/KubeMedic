#!/usr/bin/env python3
"""Standalone GRPO training script for the KubeMedic environment."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

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

from Kubemedic import KubemedicAction, KubemedicEnv

try:
    from websockets.exceptions import ConnectionClosedError
except Exception:  # pragma: no cover - optional runtime dependency path
    ConnectionClosedError = Exception


SYSTEM_PROMPT = """You are a Kubernetes SRE agent operating KubeMedic.
Diagnose the cluster carefully, use the available tools, minimize blast radius, and stop once the cluster is healthy.

Guidelines:
- Inspect before mutating.
- Prefer root-cause fixes over symptom-only actions.
- Avoid disrupting healthy running pods.
- After fixing, verify the cluster and then give a short final summary instead of continuing tool calls.
"""

TOOL_POLICY_HINT = (
    "Diagnose and repair this Kubernetes incident. "
    "Use tools carefully and finish with a concise summary when done."
)

SCENARIOS = ["KUBE-01", "KUBE-03", "KUBE-04", "KUBE-05", "KUBE-06"]
EPISODE_LOGS: list[dict[str, Any]] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a GRPO agent for the KubeMedic environment and save "
            "plots, metrics, and the final adapter."
        )
    )
    parser.add_argument(
        "--space-id",
        default="ashiqabdulkhader/Kubemedic",
        help="Hugging Face Space repo id hosting the KubeMedic environment.",
    )
    parser.add_argument(
        "--env-url",
        default=None,
        help="Optional direct environment URL. Overrides --space-id when set.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/kubemedic-qwen25-3b-grpo",
        help="Directory for checkpoints, plots, metrics, and the final adapter.",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=16,
        help="Number of training episodes. Kept small by default for a quick run.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=4,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="GRPO num_generations.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=768,
        help="Maximum generated completion length.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=4,
        help="Checkpoint save frequency.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Trainer logging frequency.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization.",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip the environment smoke test before training.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Only run the environment smoke test and exit.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_observation(obs) -> str:
    pods = []
    for pod in obs.pods:
        reason = f" reason={pod.reason}" if pod.reason else ""
        restarts = f" restarts={pod.restarts}" if pod.restarts else ""
        pods.append(f"- {pod.name}: phase={pod.phase}{reason}{restarts}")

    nodes = []
    for node in obs.nodes:
        pressure = [c.type for c in node.conditions if c.status == "True" and c.type.endswith("Pressure")]
        pressure_text = ",".join(pressure) if pressure else "none"
        nodes.append(f"- {node.name}: ready={node.ready} pressure={pressure_text}")

    info = obs.info or {}
    reward_breakdown = info.get("reward_breakdown", {})
    breakdown_text = ", ".join(f"{k}={v:.2f}" for k, v in reward_breakdown.items()) if reward_breakdown else "none"
    return (
        f"Scenario: {obs.scenario}\n"
        f"Time step: {obs.t}\n"
        f"Blocked reason: {obs.blocked_reason or 'none'}\n"
        f"Step reward: {float(obs.reward or 0.0):.2f}\n"
        f"Disruptions: {info.get('disruptions', 0)}\n"
        f"Reward breakdown: {breakdown_text}\n\n"
        f"Pods:\n" + ("\n".join(pods) if pods else "- none") + "\n\n"
        f"Nodes:\n" + ("\n".join(nodes) if nodes else "- none")
    )


def summarize_episode(log: dict[str, Any]) -> dict[str, Any]:
    tool_hist = Counter(log.get("tool_sequence", []))
    for key, value in list(tool_hist.items()):
        log[f"tool::{key}"] = value
    return log


class KubemedicToolEnv:
    def __init__(self, env_url: str, scenarios: list[str]):
        self.env_url = env_url
        self.scenarios = scenarios
        self.client = KubemedicEnv(base_url=env_url)
        self.reward = 0.0
        self.total_reward = 0.0
        self.steps = 0
        self.done = False
        self.scenario = None
        self.tool_sequence: list[str] = []
        self.last_observation = None
        self.episode_log: dict[str, Any] = {}

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

    def _run_async(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def _is_connection_error(self, exc: Exception) -> bool:
        text = str(exc)
        return (
            isinstance(exc, ConnectionClosedError)
            or "ConnectionClosed" in exc.__class__.__name__
            or "close frame" in text
            or "websocket" in text.lower()
        )

    def _is_capacity_error(self, exc: Exception) -> bool:
        text = str(exc)
        return "CAPACITY_REACHED" in text or "Server at capacity" in text or "sessions active" in text

    def _recreate_client(self) -> None:
        try:
            self._run_async(self.client.close())
        except Exception:
            pass
        self.client = KubemedicEnv(base_url=self.env_url)

    def _call_with_reconnect(self, call_factory, *, before_retry=None):
        max_attempts = 6
        for attempt in range(1, max_attempts + 1):
            try:
                return self._run_async(call_factory())
            except Exception as exc:
                is_conn_err = self._is_connection_error(exc)
                is_capacity_err = self._is_capacity_error(exc)
                if not is_conn_err and not is_capacity_err:
                    raise

                if attempt >= max_attempts:
                    raise

                delay_s = min(2 ** (attempt - 1), 20)
                if is_conn_err:
                    print(f"WebSocket dropped ({exc}); reconnecting (attempt {attempt}/{max_attempts})...")
                    self._recreate_client()
                else:
                    print(
                        f"Environment at capacity ({exc}); waiting {delay_s}s "
                        f"before retry ({attempt}/{max_attempts})..."
                    )

                if before_retry is not None and is_conn_err:
                    try:
                        before_retry()
                    except Exception as reset_exc:
                        if not (self._is_connection_error(reset_exc) or self._is_capacity_error(reset_exc)):
                            raise

                time.sleep(delay_s)

    def reset(self, prompt=None, scenario=None, **kwargs) -> str:
        del prompt, kwargs
        self.reward = 0.0
        self.total_reward = 0.0
        self.steps = 0
        self.done = False
        self.tool_sequence = []
        self.scenario = scenario or random.choice(self.scenarios)
        result = self._call_with_reconnect(lambda: self.client.reset(scenario=self.scenario))
        self.last_observation = result.observation
        self.episode_log = {
            "scenario": self.scenario,
            "steps": 0,
            "total_reward": 0.0,
            "last_step_reward": 0.0,
            "done": False,
            "solved": False,
            "disruptions": 0,
            "final_running": 0,
            "total_pods": len(self.last_observation.pods),
            "pod_restore_rate": 0.0,
            "tool_sequence": [],
            "reward_breakdown": {},
        }
        return format_observation(self.last_observation)

    def _step_tool(self, tool: str, **args) -> str:
        if self.done:
            return "Episode already ended. Give a short final answer."

        action = KubemedicAction(tool=tool, args=args)
        result = self._call_with_reconnect(
            lambda: self.client.step(action),
            before_retry=lambda: self._call_with_reconnect(lambda: self.client.reset(scenario=self.scenario)),
        )
        obs = result.observation
        self.last_observation = obs
        self.reward = float(obs.reward or 0.0)
        self.total_reward += self.reward
        self.steps += 1
        self.done = bool(obs.done)
        self.tool_sequence.append(tool)

        running = sum(1 for pod in obs.pods if pod.phase == "Running")
        total_pods = len(obs.pods)
        pod_restore_rate = (running / total_pods) if total_pods else 0.0
        info = obs.info or {}
        self.episode_log.update(
            {
                "steps": self.steps,
                "total_reward": self.total_reward,
                "last_step_reward": self.reward,
                "done": self.done,
                "solved": self.done and running == total_pods and total_pods > 0,
                "disruptions": int(info.get("disruptions", 0) or 0),
                "final_running": running,
                "total_pods": total_pods,
                "pod_restore_rate": pod_restore_rate,
                "tool_sequence": list(self.tool_sequence),
                "reward_breakdown": dict(info.get("reward_breakdown", {})),
            }
        )

        if self.done:
            EPISODE_LOGS.append(summarize_episode(dict(self.episode_log)))
            return format_observation(obs) + "\n\nCluster looks finished. Give your concise final answer now."

        return format_observation(obs)

    def kubectl_get(self, resource: str, namespace: str = "challenge", name: str | None = None) -> str:
        return self._step_tool("kubectl_get", resource=resource, namespace=namespace, name=name)

    def kubectl_describe(self, resource: str, name: str, namespace: str = "challenge") -> str:
        return self._step_tool("kubectl_describe", resource=resource, name=name, namespace=namespace)

    def kubectl_logs(self, pod_name: str, namespace: str = "challenge", previous: bool = False) -> str:
        return self._step_tool("kubectl_logs", pod_name=pod_name, namespace=namespace, previous=previous)

    def kubectl_top_pods(self, namespace: str = "challenge") -> str:
        return self._step_tool("kubectl_top_pods", namespace=namespace)

    def kubectl_top_nodes(self) -> str:
        return self._step_tool("kubectl_top_nodes")

    def kubectl_patch_resources(
        self,
        deployment_name: str,
        namespace: str,
        container_name: str,
        requests_memory_mi: int | None = None,
        limits_memory_mi: int | None = None,
        requests_cpu_m: int | None = None,
        limits_cpu_m: int | None = None,
    ) -> str:
        return self._step_tool(
            "kubectl_patch_resources",
            deployment_name=deployment_name,
            namespace=namespace,
            container_name=container_name,
            requests_memory_mi=requests_memory_mi,
            limits_memory_mi=limits_memory_mi,
            requests_cpu_m=requests_cpu_m,
            limits_cpu_m=limits_cpu_m,
        )

    def kubectl_patch_tolerations(self, deployment_name: str, namespace: str, tolerations: list[dict]) -> str:
        return self._step_tool(
            "kubectl_patch_tolerations",
            deployment_name=deployment_name,
            namespace=namespace,
            tolerations=tolerations,
        )

    def kubectl_cordon(self, node_name: str) -> str:
        return self._step_tool("kubectl_cordon", node_name=node_name)

    def kubectl_uncordon(self, node_name: str) -> str:
        return self._step_tool("kubectl_uncordon", node_name=node_name)

    def kubectl_delete_pod(self, pod_name: str, namespace: str = "challenge", force: bool = False) -> str:
        return self._step_tool("kubectl_delete_pod", pod_name=pod_name, namespace=namespace, force=force)

    def kubectl_delete_workload(self, resource: str, name: str, namespace: str = "challenge") -> str:
        return self._step_tool("kubectl_delete_workload", resource=resource, name=name, namespace=namespace)

    def close(self) -> None:
        try:
            self._run_async(self.client.close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=2)


def reward_func(environments, **kwargs) -> list[float]:
    del kwargs
    return [float(env.total_reward) for env in environments]


def build_dataset(num_rows: int, scenarios: list[str]) -> Dataset:
    rows = []
    for idx in range(num_rows):
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": TOOL_POLICY_HINT},
                ],
                "scenario": scenarios[idx % len(scenarios)],
            }
        )
    return Dataset.from_list(rows)


def make_environment_factory(env_url: str, scenarios: list[str]):
    def factory():
        return KubemedicToolEnv(env_url=env_url, scenarios=scenarios)

    return factory


def run_smoke_test(env_url: str, scenario: str) -> None:
    print(f"Connecting to {env_url} ...")
    smoke_env = KubemedicToolEnv(env_url=env_url, scenarios=SCENARIOS)
    try:
        initial = smoke_env.reset(scenario=scenario)
        print(initial[:1200])
        print("\n--- Running smoke-test tool call ---\n")
        print(smoke_env.kubectl_get(resource="pods")[:1200])
    finally:
        smoke_env.close()
    print("\nSmoke test passed.")


def save_episode_artifacts(episode_logs: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    episode_df = pd.DataFrame(episode_logs)
    if episode_df.empty:
        raise RuntimeError("No episode logs were captured. Make sure training completed at least one episode.")

    episode_df["episode"] = range(1, len(episode_df) + 1)
    episode_df["solved_int"] = episode_df["solved"].astype(int)
    episode_df["reward_ema"] = episode_df["total_reward"].ewm(span=10, adjust=False).mean()
    episode_df["solved_rate_rolling"] = episode_df["solved_int"].rolling(10, min_periods=1).mean()
    episode_df.to_csv(output_dir / "episode_metrics.csv", index=False)

    tool_cols = sorted(c for c in episode_df.columns if c.startswith("tool::"))
    tool_usage = episode_df[tool_cols].sum().sort_values(ascending=False) if tool_cols else pd.Series(dtype=float)
    scenario_stats = (
        episode_df.groupby("scenario")
        .agg(
            mean_reward=("total_reward", "mean"),
            solved_rate=("solved_int", "mean"),
            mean_steps=("steps", "mean"),
            mean_disruptions=("disruptions", "mean"),
        )
        .reset_index()
    )
    scenario_stats.to_csv(output_dir / "scenario_summary.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))

    sns.lineplot(data=episode_df, x="episode", y="total_reward", ax=axes[0, 0], label="total reward")
    sns.lineplot(data=episode_df, x="episode", y="reward_ema", ax=axes[0, 0], label="EMA(10)")
    axes[0, 0].set_title("Episode Reward")

    sns.lineplot(data=episode_df, x="episode", y="solved_rate_rolling", ax=axes[0, 1], color="green")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("Rolling Solved Rate (10 episodes)")

    sns.lineplot(data=episode_df, x="episode", y="steps", ax=axes[1, 0], label="steps")
    sns.lineplot(data=episode_df, x="episode", y="disruptions", ax=axes[1, 0], label="disruptions")
    axes[1, 0].set_title("Steps and Disruptions")

    sns.barplot(data=scenario_stats, x="scenario", y="mean_reward", ax=axes[1, 1], palette="crest")
    axes[1, 1].set_title("Mean Reward by Scenario")

    sns.barplot(data=scenario_stats, x="scenario", y="solved_rate", ax=axes[2, 0], palette="viridis")
    axes[2, 0].set_ylim(0, 1.05)
    axes[2, 0].set_title("Solved Rate by Scenario")

    if not tool_usage.empty:
        sns.barplot(x=tool_usage.values, y=tool_usage.index, ax=axes[2, 1], palette="magma")
        axes[2, 1].set_title("Tool Usage Frequency")
    else:
        axes[2, 1].text(0.5, 0.5, "No tool usage captured", ha="center", va="center")
        axes[2, 1].set_title("Tool Usage Frequency")

    plt.tight_layout()
    fig.savefig(output_dir / "episode_dashboard.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(episode_df["total_reward"], bins=20, kde=True, ax=axes[0])
    axes[0].set_title("Reward Distribution")

    sns.scatterplot(data=episode_df, x="steps", y="total_reward", hue="scenario", ax=axes[1])
    axes[1].set_title("Reward vs Steps")

    sns.scatterplot(data=episode_df, x="pod_restore_rate", y="total_reward", hue="solved", ax=axes[2])
    axes[2].set_title("Reward vs Pod Restore Rate")
    plt.tight_layout()
    fig.savefig(output_dir / "episode_scatterplots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return episode_df


def save_trainer_artifacts(trainer: GRPOTrainer, output_dir: Path) -> pd.DataFrame:
    train_history = pd.DataFrame(trainer.state.log_history)
    train_history.to_csv(output_dir / "trainer_log_history.csv", index=False)

    if train_history.empty:
        return train_history

    numeric_cols = [c for c in ["loss", "grad_norm", "learning_rate"] if c in train_history.columns]
    if numeric_cols:
        fig, axes = plt.subplots(1, len(numeric_cols), figsize=(6 * len(numeric_cols), 4))
        if len(numeric_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, numeric_cols):
            filtered = train_history.dropna(subset=[col])
            if filtered.empty or "step" not in filtered.columns:
                continue
            sns.lineplot(data=filtered, x="step", y=col, ax=ax)
            ax.set_title(f"Trainer {col}")
        plt.tight_layout()
        fig.savefig(output_dir / "trainer_metrics.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    return train_history


def write_summary(
    args: argparse.Namespace,
    output_dir: Path,
    episode_df: pd.DataFrame | None,
    train_history: pd.DataFrame | None,
    adapter_dir: Path | None,
) -> None:
    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "env_url": resolve_env_url(args),
        "output_dir": str(output_dir),
        "train_episodes": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "seed": args.seed,
        "use_4bit": not args.no_4bit,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "artifacts": sorted(path.name for path in output_dir.iterdir()),
    }

    if episode_df is not None and not episode_df.empty:
        summary["episodes_logged"] = int(len(episode_df))
        summary["mean_reward"] = float(episode_df["total_reward"].mean())
        summary["mean_steps"] = float(episode_df["steps"].mean())
        summary["solved_rate"] = float(episode_df["solved_int"].mean())

    if train_history is not None and not train_history.empty:
        latest = train_history.dropna(how="all").tail(1).to_dict(orient="records")
        summary["last_trainer_log"] = latest[0] if latest else {}

    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))


def resolve_env_url(args: argparse.Namespace) -> str:
    if args.env_url:
        return args.env_url
    return f"https://{args.space_id.replace('/', '-')}.hf.space"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    sns.set_theme(style="whitegrid")
    pd.set_option("display.max_colwidth", 120)

    env_url = resolve_env_url(args)
    if not args.skip_smoke_test:
        run_smoke_test(env_url=env_url, scenario="KUBE-03")
    if args.smoke_only:
        return

    EPISODE_LOGS.clear()

    train_dataset = build_dataset(args.train_episodes, SCENARIOS)
    eval_dataset = build_dataset(args.eval_episodes, SCENARIOS)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quant_config,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
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
        report_to=[],
        log_completions=True,
        seed=args.seed,
    )

    if Version(transformers.__version__) < Version("5.2.0"):
        import trl.trainer.grpo_trainer as _grpo_trainer

        print(
            "Applying compatibility shim for TRL environment_factory "
            f"(detected transformers {transformers.__version__})."
        )
        transformers.__version__ = "5.2.0"
        _grpo_trainer.transformers.__version__ = "5.2.0"

    if getattr(tokenizer, "response_schema", None) is None:
        import trl.chat_template_utils as _ctu

        tokenizer.response_schema = _ctu.qwen3_schema
        print("Applied manual tokenizer.response_schema (qwen3_schema).")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=grpo_config,
        environment_factory=make_environment_factory(env_url=env_url, scenarios=SCENARIOS),
        peft_config=peft_config,
    )

    print("Trainer initialized.")
    train_result = trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    episode_df = save_episode_artifacts(EPISODE_LOGS, output_dir)
    train_history = save_trainer_artifacts(trainer, output_dir)
    write_summary(args, output_dir, episode_df, train_history, adapter_dir)

    print(train_result)
    print(f"Saved adapter to {adapter_dir}")
    print(f"Saved graphs and metrics to {output_dir}")


if __name__ == "__main__":
    main()
