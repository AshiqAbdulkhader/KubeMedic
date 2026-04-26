"""KubeMedic GRPO training (self-contained, no parameters).

Run it directly with `python training.py`. The script:
  1. Ensures system deps (clones the KubeMedic Space, pins trl/datasets, installs unsloth).
  2. Loads `.env`, configures wandb, and primes A100-friendly torch backends.
  3. Loads `unsloth/gemma-4-E2B-it` (with vLLM + LoRA via unsloth) and runs GRPO for 5 epochs.
  4. Stores all results under `outputs/kubemedic-gemma-4-E2B-grpo/`:
       - `episode_rewards.csv`, `agent_transcripts.jsonl`, `trainer_log_history.csv`
       - `reward_plot.png`, `trainer_metrics.png`, `training_summary.json`
       - `adapter/` (LoRA weights + tokenizer)
     and pushes a wandb run under `kubemedic-grpo-gemma-4-E2B`.

Skip the install pass on subsequent runs by exporting `KUBEMEDIC_SKIP_INSTALL=1`.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# 1. Dependency bootstrap (cell 2 in the notebook).                           #
# --------------------------------------------------------------------------- #

SPACE_ID = "ashiqabdulkhader/Kubemedic"
SPACE_URL = f"https://huggingface.co/spaces/{SPACE_ID}.git"
KUBEMEDIC_SRC = "/tmp/kubemedic-src"
WANDB_API_KEY_LITERAL = (
    "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"
)


def _pip(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    print("$", " ".join(shlex.quote(a) for a in cmd))
    subprocess.check_call(cmd)


def _ensure_dependencies() -> None:
    if os.environ.get("KUBEMEDIC_SKIP_INSTALL", "").strip() in {"1", "true", "TRUE"}:
        print("KUBEMEDIC_SKIP_INSTALL set; skipping dependency bootstrap.")
        return

    if os.path.isdir(KUBEMEDIC_SRC):
        shutil.rmtree(KUBEMEDIC_SRC)
    print("Cloning Space:", SPACE_URL)
    subprocess.check_call(["git", "clone", "--depth", "1", SPACE_URL, KUBEMEDIC_SRC])

    _pip("-U", "numpy>=2.0,<3")
    _pip("-U", "wandb>=0.17.0", "nbformat>=5.10", "hf_transfer>=0.1.6")
    _pip("-U", f"{KUBEMEDIC_SRC}[train]")
    _pip("--force-reinstall", "trl>=0.18.2,<=0.24.0", "datasets>=3.4.1,<4.4.0")
    _pip("-U", "--no-deps", "--force-reinstall", "unsloth", "unsloth_zoo")

    cache_dir = os.path.expanduser("~/unsloth_compiled_cache")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        print("Cleared ~/unsloth_compiled_cache so GRPO patches recompile against pinned trl.")


_ensure_dependencies()


# --------------------------------------------------------------------------- #
# 2. Environment setup (cell 4 in the notebook).                              #
# --------------------------------------------------------------------------- #

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kubemedic-mpl"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ.setdefault("WANDB_DISABLED", "false")
os.environ.setdefault("WANDB_NOTEBOOK_NAME", "kubemedic_training.py")
os.environ.setdefault("WANDB_SAVE_CODE", "false")
os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY") or WANDB_API_KEY_LITERAL

for env_path in (SCRIPT_DIR / ".env", SCRIPT_DIR.parent / ".env"):
    if not env_path.exists():
        continue
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

if not os.environ.get("HF_TOKEN"):
    try:
        os.environ["HF_TOKEN"] = subprocess.check_output(["hf", "auth", "token"], text=True).strip()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# 3. Unsloth import — must happen before torch / trl / transformers.          #
# --------------------------------------------------------------------------- #

try:
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]

    UNSLOTH_AVAILABLE = True
    try:
        from unsloth import FastModel  # type: ignore[import-not-found]
    except ImportError:
        FastModel = None
    try:
        from unsloth import PatchFastRL  # type: ignore[import-not-found]

        PatchFastRL("GRPO", FastLanguageModel)
        print("unsloth loaded; GRPO patches applied via PatchFastRL.")
    except ImportError:
        print("unsloth loaded; relying on unsloth_zoo auto-patches (no PatchFastRL).")
    except Exception as patch_exc:
        print(f"unsloth loaded; PatchFastRL skipped ({patch_exc}); auto-patches still active.")
except Exception as exc:
    print(f"unsloth unavailable ({exc}); falling back to standard transformers loading.")
    FastLanguageModel = None
    FastModel = None
    UNSLOTH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# 4. Heavy imports.                                                            #
# --------------------------------------------------------------------------- #

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
import wandb  # noqa: E402
from datasets import Dataset  # noqa: E402
from packaging.version import Version  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

try:
    from trl.experimental.openenv import generate_rollout_completions  # noqa: E402
except ImportError:
    generate_rollout_completions = None  # noqa: E402 — set after local def in main

from Kubemedic import KubemedicAction, KubemedicEnv, KubemedicObservation  # noqa: E402
from Kubemedic.models import ToolName  # noqa: E402

try:
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK  # noqa: E402
except Exception:
    ConnectionClosedError = Exception
    ConnectionClosedOK = Exception


# --------------------------------------------------------------------------- #
# 5. Configuration (cell 6).                                                  #
# --------------------------------------------------------------------------- #

ENV_URL = f"https://{SPACE_ID.replace('/', '-')}.hf.space"
MODEL_ID = "unsloth/gemma-4-E2B-it"
OUTPUT_DIR = Path("outputs/kubemedic-gemma-4-E2B-grpo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_EPISODES = 16
EVAL_EPISODES = 2
NUM_EPOCHS = 5
NUM_GENERATIONS = 4
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-5
MAX_COMPLETION_LENGTH = 512
MAX_TURNS = 8
SAVE_STEPS = 4
LOGGING_STEPS = 1
SEED = 42
VLLM_MODE = "colocate"
VLLM_GPU_MEMORY_UTILIZATION = 0.5
USE_4BIT = False
USE_VLLM = True
USE_UNSLOTH = True
USE_GRADIENT_CHECKPOINTING = False
USE_FLASH_ATTENTION = True
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
SKIP_SMOKE_TEST = False
WANDB_PROJECT = "kubemedic-grpo-gemma-4-E2B"
WANDB_RUN_NAME = None
SCENARIOS = ["KUBE-01", "KUBE-03", "KUBE-04", "KUBE-05", "KUBE-06"]

CONNECT_TIMEOUT_S = 30.0
MESSAGE_TIMEOUT_S = 240.0
VALID_TOOLS = set(ToolName.__args__)  # type: ignore[attr-defined]

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


# --------------------------------------------------------------------------- #
# 6. Helper functions (cell 8).                                               #
# --------------------------------------------------------------------------- #


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_wandb(project: str, run_name: str | None) -> list[str]:
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not api_key:
        os.environ["WANDB_DISABLED"] = "true"
        print("WANDB_API_KEY not set; wandb disabled.")
        return []
    os.environ["WANDB_DISABLED"] = "false"
    os.environ.setdefault("WANDB_PROJECT", project)
    if run_name:
        os.environ.setdefault("WANDB_NAME", run_name)
    try:
        wandb.login(key=api_key, relogin=True)
        msg = f"wandb authenticated; project={os.environ['WANDB_PROJECT']}"
        if os.environ.get("WANDB_NAME"):
            msg += f" run={os.environ['WANDB_NAME']}"
        print(msg)
    except Exception as exc:
        print("wandb.login failed:", exc, "-- disabling wandb for this run.")
        os.environ["WANDB_DISABLED"] = "true"
        return []
    return ["wandb"]


def make_env(env_url: str):
    return KubemedicEnv(
        base_url=env_url,
        connect_timeout_s=CONNECT_TIMEOUT_S,
        message_timeout_s=MESSAGE_TIMEOUT_S,
    ).sync()


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
    breakdown_text = (
        ", ".join(f"{k}={v:.2f}" for k, v in reward_breakdown.items()) if reward_breakdown else "none"
    )

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
    if lowered in ("none", "null"):
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
    line = cleaned.splitlines()[0].strip() if cleaned else ""
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


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
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


def build_dataset(num_rows: int) -> Dataset:
    rows = [{"prompt": TRAINING_HINT} for _ in range(num_rows)]
    return Dataset.from_list(rows)


def infer_lora_target_modules(model) -> list[str]:
    model_type = getattr(model.config, "model_type", "")
    if model_type in {"qwen2", "qwen2_5", "qwen3"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if model_type in {"gpt2"}:
        return ["c_attn", "c_proj", "c_fc"]
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    preferred = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "c_attn", "c_proj", "c_fc",
    ]
    inferred = [name for name in preferred if name in module_names]
    if inferred:
        return inferred
    raise ValueError(f"Could not infer LoRA target modules for model_type={model_type!r}.")


def reward_total(completions: list[str], **kwargs) -> list[float]:
    rewards = kwargs.get("total_reward") if kwargs else None
    return [float(value) for value in rewards] if rewards else [0.0 for _ in completions]


# --------------------------------------------------------------------------- #
# 7. Rollout helpers (cell 10).                                               #
# --------------------------------------------------------------------------- #


def generate_rollout_completions_local(trainer: GRPOTrainer, prompts: list[str]) -> list[dict[str, Any]]:
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


if generate_rollout_completions is None:
    generate_rollout_completions = generate_rollout_completions_local
    if USE_VLLM:
        import warnings

        warnings.warn(
            "trl.experimental.openenv is not installed: vLLM rollouts will use local HF `model.generate` "
            "instead. Install a trl build that provides trl.experimental.openenv, or set USE_VLLM=False.",
            stacklevel=1,
        )


def rollout_once(
    trainer: GRPOTrainer,
    env_url: str,
    tokenizer,
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
            valid_actions = 0
            total_actions = 0
            tool_usage: dict[str, int] = {}
            sample_completion = ""

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
                if not sample_completion:
                    sample_completion = completion_text[:600]
                total_actions += 1
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

                valid_actions += 1
                tool_usage[action.tool] = tool_usage.get(action.tool, 0) + 1
                result = env.step(action)
                observation = result.observation
                reward = float(result.reward or 0.0)
                step_rewards.append(reward)
                tool_output = json.dumps(observation.tool_result or {}, default=str)
                history.append({"action": completion_text, "output": tool_output, "reward": reward})

            total_reward = sum(step_rewards) if step_rewards else -1.0
            mean_step_reward = (total_reward / len(step_rewards)) if step_rewards else 0.0
            min_step_reward = min(step_rewards) if step_rewards else 0.0
            max_step_reward = max(step_rewards) if step_rewards else 0.0
            valid_action_rate = (valid_actions / total_actions) if total_actions else 0.0
            mean_logprob = (sum(logprobs) / len(logprobs)) if logprobs else 0.0
            return {
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "logprobs": logprobs,
                "total_reward": total_reward,
                "mean_step_reward": mean_step_reward,
                "min_step_reward": min_step_reward,
                "max_step_reward": max_step_reward,
                "mean_logprob": mean_logprob,
                "completion_tokens": len(completion_ids),
                "prompt_tokens": len(prompt_ids),
                "steps": len(step_rewards),
                "scenario": scenario,
                "resolved": bool(result.done),
                "valid_actions": valid_actions,
                "total_actions": total_actions,
                "valid_action_rate": valid_action_rate,
                "tool_usage": tool_usage,
                "sample_completion": sample_completion,
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


# --------------------------------------------------------------------------- #
# 8. Smoke test (cell 12).                                                    #
# --------------------------------------------------------------------------- #


def smoke_test(env_url: str, scenario: str) -> None:
    env = make_env(env_url)
    try:
        result = env.reset(scenario=scenario)
        print(format_observation(result.observation)[:1200])
        step_result = env.step(
            KubemedicAction(tool="kubectl_get", args={"resource": "pods", "namespace": "challenge"})
        )
        print("\n--- smoke tool call ---\n")
        print(format_observation(step_result.observation)[:1200])
        print("\nSmoke test passed.")
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# 9. Plotting (cell 18).                                                      #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# 10. wandb safety helpers (cell 16).                                         #
# --------------------------------------------------------------------------- #

_WANDB_STATE = {"warned": False, "broken": False}


@contextlib.contextmanager
def _wandb_guard(label: str = "log"):
    """Swallow any wandb exception so it never crashes training."""
    if wandb is None or _WANDB_STATE["broken"]:
        yield
        return
    try:
        yield
    except Exception as exc:
        if not _WANDB_STATE["warned"]:
            print(f"[wandb] {label} failed: {exc}. Suppressing further wandb errors so training can continue.")
            _WANDB_STATE["warned"] = True
        if isinstance(exc, (RuntimeError, OSError)):
            _WANDB_STATE["broken"] = True


def _wandb_active() -> bool:
    if wandb is None or _WANDB_STATE["broken"]:
        return False
    try:
        return wandb.run is not None
    except Exception:
        return False


def _strip_wandb_callback(trainer: GRPOTrainer) -> None:
    handler = getattr(trainer, "callback_handler", None)
    if handler is None:
        return
    handler.callbacks = [cb for cb in handler.callbacks if cb.__class__.__name__ != "WandbCallback"]


def _wrap_callback_method(trainer: GRPOTrainer, method_name: str) -> None:
    handler = getattr(trainer, "callback_handler", None)
    if handler is None:
        return
    for cb in list(handler.callbacks):
        if cb.__class__.__name__ != "WandbCallback":
            continue
        original = getattr(cb, method_name, None)
        if not callable(original):
            continue

        def _safe(*args, _orig=original, **kwargs):
            with _wandb_guard(f"callback.{method_name}"):
                return _orig(*args, **kwargs)
            if _WANDB_STATE["broken"]:
                _strip_wandb_callback(trainer)

        setattr(cb, method_name, _safe)


def _safe_wandb_finish() -> None:
    try:
        if wandb.run is not None:
            wandb.finish(exit_code=1, quiet=True)
    except Exception as exc:
        print(f"[wandb] finish failed (ignored): {exc}")


# --------------------------------------------------------------------------- #
# 11. main — runs the full pipeline end-to-end (cells 14, 16, 18, 20).        #
# --------------------------------------------------------------------------- #


def main() -> None:
    print("=== KubeMedic GRPO training (script mode) ===")
    print("Kernel python   :", sys.executable)
    print("Torch version   :", torch.__version__)
    print("CUDA available  :", torch.cuda.is_available())
    print("Unsloth         :", UNSLOTH_AVAILABLE)
    print("HF token loaded :", bool(os.environ.get("HF_TOKEN")))
    print("WANDB key loaded:", bool(os.environ.get("WANDB_API_KEY")))
    print("ENV_URL   :", ENV_URL)
    print("MODEL_ID  :", MODEL_ID)
    print("OUTPUT_DIR:", OUTPUT_DIR.resolve())
    print("SCENARIOS :", SCENARIOS)

    # ----------- preflight ---------------------------------------------------
    if PER_DEVICE_TRAIN_BATCH_SIZE % NUM_GENERATIONS != 0:
        raise ValueError("PER_DEVICE_TRAIN_BATCH_SIZE must be divisible by NUM_GENERATIONS for GRPO.")
    if Version(transformers.__version__) < Version("4.56.0"):
        raise RuntimeError("transformers>=4.56.0 is required for the training entrypoint.")

    set_seed(SEED)
    report_to = configure_wandb(WANDB_PROJECT, WANDB_RUN_NAME)

    if not SKIP_SMOKE_TEST:
        smoke_test(ENV_URL, SCENARIOS[0])
    else:
        print("Skipping smoke test.")

    # ----------- A100 backend tuning ----------------------------------------
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    use_vllm = torch.cuda.is_available() and USE_VLLM
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    use_unsloth = bool(USE_UNSLOTH and UNSLOTH_AVAILABLE and torch.cuda.is_available())
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    attn_impl = "sdpa" if USE_FLASH_ATTENTION else "eager"

    # ----------- model + tokenizer ------------------------------------------
    model = None
    tokenizer = None
    peft_config: LoraConfig | None = None
    quant_config = None

    if use_unsloth:
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
            try:
                import hf_transfer  # noqa: F401
            except Exception:
                print("hf_transfer not importable; disabling HF_HUB_ENABLE_HF_TRANSFER for this run.")
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        use_fast_model = ("gemma" in MODEL_ID.lower() and FastModel is not None)
        loader = FastModel if use_fast_model else FastLanguageModel
        from_pretrained_kwargs: dict[str, Any] = dict(
            model_name=MODEL_ID,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=dtype,
            load_in_4bit=USE_4BIT,
        )
        if use_vllm:
            from_pretrained_kwargs.update(
                fast_inference=True,
                max_lora_rank=LORA_RANK,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            )
        try:
            try:
                model, tokenizer = loader.from_pretrained(**from_pretrained_kwargs)
            except TypeError as kwarg_exc:
                if use_vllm:
                    print(
                        f"unsloth loader does not accept fast_inference kwargs ({kwarg_exc}); "
                        "retrying without vLLM fast-inference."
                    )
                    for k in ("fast_inference", "max_lora_rank", "gpu_memory_utilization"):
                        from_pretrained_kwargs.pop(k, None)
                    model, tokenizer = loader.from_pretrained(**from_pretrained_kwargs)
                    use_vllm = False
                else:
                    raise
            peft_kwargs: dict[str, Any] = dict(
                r=LORA_RANK,
                target_modules=lora_target_modules,
                lora_alpha=LORA_RANK * 2,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing=("unsloth" if USE_GRADIENT_CHECKPOINTING else False),
                random_state=SEED,
                use_rslora=False,
                loftq_config=None,
            )
            if use_fast_model:
                peft_kwargs.update(
                    finetune_vision_layers=False,
                    finetune_language_layers=True,
                    finetune_attention_modules=True,
                    finetune_mlp_modules=True,
                )
            model = loader.get_peft_model(model, **peft_kwargs)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("Loader    :", "unsloth.FastModel" if use_fast_model else "unsloth.FastLanguageModel")
            print("LoRA tgts :", lora_target_modules)
        except Exception as exc:
            print(f"unsloth loader failed at runtime ({exc}); falling back to standard transformers path.")
            use_unsloth = False
            model = None
            tokenizer = None

    if not use_unsloth:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if torch.cuda.is_available() and USE_4BIT:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )

        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "quantization_config": quant_config,
            "attn_implementation": attn_impl,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=infer_lora_target_modules(model),
        )
        print("Loader    : transformers.AutoModelForCausalLM")
        print("LoRA tgts :", peft_config.target_modules)

    print("use_vllm  :", use_vllm)
    print("use_unsloth:", use_unsloth)
    print("attn impl :", attn_impl)
    print("dtype     :", dtype)
    print("4-bit     :", USE_4BIT and torch.cuda.is_available())
    print("grad ckpt :", USE_GRADIENT_CHECKPOINTING)

    # ----------- datasets + GRPOConfig --------------------------------------
    train_dataset = build_dataset(TRAIN_EPISODES)
    eval_dataset = build_dataset(EVAL_EPISODES)

    trainer_grad_ckpt = bool(USE_GRADIENT_CHECKPOINTING and not use_unsloth)
    grpo_config_kwargs = dict(
        output_dir=str(OUTPUT_DIR),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_train_epochs=NUM_EPOCHS,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        save_total_limit=2,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        tf32=torch.cuda.is_available(),
        gradient_checkpointing=trainer_grad_ckpt,
        remove_unused_columns=False,
        report_to=report_to,
        use_vllm=use_vllm,
        vllm_mode=VLLM_MODE,
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        log_completions=True,
        max_tool_calling_iterations=1,
        seed=SEED,
        optim="adamw_torch_fused",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )
    try:
        grpo_config = GRPOConfig(**grpo_config_kwargs)
    except TypeError as exc:
        print(f"GRPOConfig rejected an A100 kwarg ({exc}); retrying with the safe subset.")
        for k in ("tf32", "optim", "dataloader_num_workers", "dataloader_pin_memory"):
            grpo_config_kwargs.pop(k, None)
        grpo_config = GRPOConfig(**grpo_config_kwargs)

    # ----------- result file scaffolding ------------------------------------
    reward_log_path = OUTPUT_DIR / "episode_rewards.csv"
    transcript_path = OUTPUT_DIR / "agent_transcripts.jsonl"
    if transcript_path.exists():
        transcript_path.unlink()
    with reward_log_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "scenario", "total_reward", "steps", "resolved"])

    # ----------- wandb run + episode table ----------------------------------
    if wandb is not None and "wandb" in report_to and not _wandb_active():
        with _wandb_guard("init"):
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT),
                name=os.environ.get("WANDB_NAME", WANDB_RUN_NAME),
                config={
                    "model_id": MODEL_ID,
                    "env_url": ENV_URL,
                    "train_episodes": TRAIN_EPISODES,
                    "eval_episodes": EVAL_EPISODES,
                    "num_epochs": NUM_EPOCHS,
                    "num_generations": NUM_GENERATIONS,
                    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
                    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                    "learning_rate": LEARNING_RATE,
                    "max_completion_length": MAX_COMPLETION_LENGTH,
                    "max_turns": MAX_TURNS,
                    "use_vllm": use_vllm,
                    "vllm_mode": VLLM_MODE,
                    "vllm_gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
                    "use_4bit": USE_4BIT,
                    "use_unsloth": use_unsloth,
                    "use_gradient_checkpointing": bool(USE_GRADIENT_CHECKPOINTING),
                    "attn_implementation": attn_impl,
                    "lora_rank": LORA_RANK,
                    "seed": SEED,
                    "scenarios": SCENARIOS,
                },
                tags=["kubemedic", "grpo", "gemma-4", "gemma-4-E2B"],
                save_code=False,
                reinit="finish_previous",
            )
        if _wandb_active():
            try:
                print("wandb run:", wandb.run.url)
            except Exception:
                print("wandb run: (active, url unavailable)")
        else:
            print("wandb run: (disabled)")

    episode_counter = {"value": 0}
    episode_table = None
    if _wandb_active():
        with _wandb_guard("Table.init"):
            episode_table = wandb.Table(
                columns=[
                    "episode", "scenario", "total_reward", "mean_step_reward",
                    "min_step_reward", "max_step_reward", "steps", "resolved",
                    "valid_action_rate", "completion_tokens", "prompt_tokens",
                    "mean_logprob", "top_tool", "sample_completion",
                ]
            )

    roll_window = 32
    recent_rewards: list[float] = []
    recent_resolved: list[int] = []
    recent_steps: list[int] = []
    recent_valid_rate: list[float] = []
    scenario_rewards: dict[str, list[float]] = {s: [] for s in SCENARIOS}

    def running_mean(values: list[float], window: int) -> float:
        if not values:
            return 0.0
        tail = values[-window:]
        return sum(tail) / len(tail)

    # ----------- rollout func -----------------------------------------------
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        prompt_count = len(prompts)
        episode_prompt_ids: list[list[int]] = []
        episode_completion_ids: list[list[int]] = []
        episode_logprobs: list[list[float]] = []
        total_rewards: list[float] = []
        steps: list[int] = []
        scenario_names: list[str] = []
        resolved_flags: list[bool] = []

        for start in range(0, prompt_count, NUM_GENERATIONS):
            scenario = random.choice(SCENARIOS)
            group_size = min(NUM_GENERATIONS, prompt_count - start)
            for _ in range(group_size):
                episode = rollout_once(
                    trainer=trainer,
                    env_url=ENV_URL,
                    tokenizer=tokenizer,
                    scenario=scenario,
                    max_turns=MAX_TURNS,
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

                recent_rewards.append(float(episode["total_reward"]))
                recent_resolved.append(int(bool(episode["resolved"])))
                recent_steps.append(int(episode["steps"]))
                recent_valid_rate.append(float(episode["valid_action_rate"]))
                scenario_rewards.setdefault(episode["scenario"], []).append(
                    float(episode["total_reward"])
                )

                if _wandb_active():
                    scenario_index = (
                        SCENARIOS.index(episode["scenario"])
                        if episode["scenario"] in SCENARIOS
                        else -1
                    )
                    tool_usage = episode.get("tool_usage") or {}
                    top_tool = max(tool_usage.items(), key=lambda kv: kv[1])[0] if tool_usage else ""

                    log_payload = {
                        "rollout/episode": episode_counter["value"],
                        "rollout/total_reward": float(episode["total_reward"]),
                        "rollout/mean_step_reward": float(episode["mean_step_reward"]),
                        "rollout/min_step_reward": float(episode["min_step_reward"]),
                        "rollout/max_step_reward": float(episode["max_step_reward"]),
                        "rollout/mean_logprob": float(episode["mean_logprob"]),
                        "rollout/steps": int(episode["steps"]),
                        "rollout/resolved": int(bool(episode["resolved"])),
                        "rollout/scenario_index": scenario_index,
                        "rollout/valid_action_rate": float(episode["valid_action_rate"]),
                        "rollout/valid_actions": int(episode["valid_actions"]),
                        "rollout/total_actions": int(episode["total_actions"]),
                        "rollout/completion_tokens": int(episode["completion_tokens"]),
                        "rollout/prompt_tokens": int(episode["prompt_tokens"]),
                        f"rollout/scenario_reward/{episode['scenario']}": float(episode["total_reward"]),
                        f"rollout/scenario_count/{episode['scenario']}": len(scenario_rewards[episode["scenario"]]),
                        "running/mean_reward": running_mean(recent_rewards, roll_window),
                        "running/resolution_rate": running_mean(
                            [float(v) for v in recent_resolved], roll_window
                        ),
                        "running/mean_steps": running_mean([float(v) for v in recent_steps], roll_window),
                        "running/valid_action_rate": running_mean(recent_valid_rate, roll_window),
                        "rollout/cumulative_resolved": int(sum(recent_resolved)),
                        "rollout/cumulative_episodes": episode_counter["value"],
                    }
                    for tool_name, count in tool_usage.items():
                        log_payload[f"tool_usage/{tool_name}"] = int(count)

                    with _wandb_guard("log"):
                        wandb.log(log_payload)

                    if episode_table is not None:
                        with _wandb_guard("Table.add_data"):
                            episode_table.add_data(
                                episode_counter["value"],
                                episode["scenario"],
                                float(episode["total_reward"]),
                                float(episode["mean_step_reward"]),
                                float(episode["min_step_reward"]),
                                float(episode["max_step_reward"]),
                                int(episode["steps"]),
                                bool(episode["resolved"]),
                                float(episode["valid_action_rate"]),
                                int(episode["completion_tokens"]),
                                int(episode["prompt_tokens"]),
                                float(episode["mean_logprob"]),
                                top_tool,
                                episode.get("sample_completion", ""),
                            )

        return {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs": episode_logprobs,
            "total_reward": total_rewards,
            "steps": steps,
            "scenario": scenario_names,
            "resolved": resolved_flags,
        }

    # ----------- trainer init -----------------------------------------------
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

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

    if _WANDB_STATE["broken"] or wandb is None:
        _strip_wandb_callback(trainer)
    else:
        for method in (
            "on_init_end", "on_train_begin", "on_log", "on_save",
            "on_evaluate", "on_predict", "on_train_end",
        ):
            _wrap_callback_method(trainer, method)

    # ----------- train ------------------------------------------------------
    try:
        trainer.train()
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        if "wandb" in msg.lower() and not _WANDB_STATE["broken"]:
            print(f"[wandb] training raised {msg}; disabling wandb and resuming.")
            _WANDB_STATE["broken"] = True
            _strip_wandb_callback(trainer)
            os.environ["WANDB_DISABLED"] = "true"
            if hasattr(trainer, "args") and getattr(trainer.args, "report_to", None):
                trainer.args.report_to = [r for r in trainer.args.report_to if r != "wandb"]
            _safe_wandb_finish()
            trainer.train()
        else:
            raise

    # ----------- artifacts + plots ------------------------------------------
    adapter_dir = OUTPUT_DIR / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    reward_plot_path = OUTPUT_DIR / "reward_plot.png"
    trainer_metrics_path = OUTPUT_DIR / "trainer_metrics.png"
    log_history_path = OUTPUT_DIR / "trainer_log_history.csv"
    summary_path = OUTPUT_DIR / "training_summary.json"

    plot_rewards(reward_log_path, reward_plot_path)
    log_history = pd.DataFrame(trainer.state.log_history)
    log_history.to_csv(log_history_path, index=False)
    plot_trainer_metrics(log_history, trainer_metrics_path)

    summary = {
        "model_id": MODEL_ID,
        "env_url": ENV_URL,
        "train_episodes": TRAIN_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "num_generations": NUM_GENERATIONS,
        "num_epochs": NUM_EPOCHS,
        "output_dir": str(OUTPUT_DIR),
        "adapter_dir": str(adapter_dir),
    }
    wandb_run_url = None
    if _wandb_active():
        try:
            wandb_run_url = wandb.run.url
        except Exception:
            wandb_run_url = None
    summary["wandb_run_url"] = wandb_run_url
    summary_path.write_text(json.dumps(summary, indent=2))
    print("Wrote artifacts under", OUTPUT_DIR.resolve())

    if _wandb_active():
        if reward_plot_path.exists():
            with _wandb_guard("log reward_curve"):
                wandb.log({"plots/reward_curve": wandb.Image(str(reward_plot_path))})
        if trainer_metrics_path.exists():
            with _wandb_guard("log trainer_metrics"):
                wandb.log({"plots/trainer_metrics": wandb.Image(str(trainer_metrics_path))})
        if episode_table is not None:
            with _wandb_guard("log episode_table"):
                wandb.log({"rollout/episode_table": episode_table})

        run_artifact = None
        with _wandb_guard("Artifact run"):
            run_artifact = wandb.Artifact("kubemedic-grpo-run", type="training-output")
            for path in (
                reward_log_path, log_history_path, summary_path,
                reward_plot_path, trainer_metrics_path, transcript_path,
            ):
                if path.exists():
                    run_artifact.add_file(str(path))
        if run_artifact is not None:
            with _wandb_guard("log_artifact run"):
                wandb.log_artifact(run_artifact)

        if adapter_dir.exists():
            adapter_artifact = None
            with _wandb_guard("Artifact adapter"):
                adapter_artifact = wandb.Artifact(
                    "kubemedic-grpo-adapter",
                    type="model",
                    metadata={"base_model": MODEL_ID, "kind": "lora"},
                )
                adapter_artifact.add_dir(str(adapter_dir))
            if adapter_artifact is not None:
                with _wandb_guard("log_artifact adapter"):
                    wandb.log_artifact(adapter_artifact)

        with _wandb_guard("finish"):
            wandb.finish()
        print("wandb run finished (artifact upload best-effort).")
    else:
        print("wandb not active; skipping wandb artifact upload.")

    # ----------- final result echo (cell 20) --------------------------------
    print("\n=== Final results ===")
    if reward_log_path.exists():
        print(pd.read_csv(reward_log_path).tail(10).to_string(index=False))
    print(json.dumps(summary, indent=2))
    print("Reward plot   :", reward_plot_path)
    print("Trainer plot  :", trainer_metrics_path)
    print("Adapter dir   :", adapter_dir)


if __name__ == "__main__":
    main()
