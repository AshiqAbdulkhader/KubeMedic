# Every On-Call Shift is an Episode. I Just Hadn't Built the Reward Function Yet.

It's 2:47 AM. My phone is going off. Not a call — a cascade of them.

I'm a DevSecOps engineer. I live between developers shipping fast and operations keeping things from catching fire. But this particular night I'm sitting in bed, laptop open, Slack exploding, with that specific dread you only feel when you already know what's broken before you've typed a single command.

The customer portal is down.

> *"Move fast and break things"* — everybody quotes Zuckerberg on this. Nobody quotes what comes after: when you break something in production, you better have someone who knows how to unbreak it faster.

The culprit was a Kubernetes cluster in a bad state. A node in `MemoryPressure`, pods evicting, the payment service stuck in `CrashLoopBackOff`. Ninety minutes of manual diagnosis — pulling logs, cross-referencing resource limits, patching a deployment — and I fixed it. But I didn't sleep after.

Every minute that portal was dark, someone was deciding whether to trust us with their business. And I kept thinking: *what if the next engineer on-call doesn't know what I know? What if I'd made the wrong call at 3 AM?*

Here's the thing about Kubernetes incident response that no conference talk covers: **you can't practice it**. Staging doesn't have real memory pressure. Sandboxes don't have the same 3 AM energy where your hands are slightly shaking and you're second-guessing every `kubectl delete`. We give junior engineers toy environments with mock APIs, then wonder why their first real on-call shift is brutal.

> *"Creativity is just connecting things."* — Steve Jobs. The best SREs I know aren't the ones who memorized runbooks. They're the ones who can connect what they're seeing in logs to what's actually failing in the cluster. That's a skill, and you don't build it by reading about it.

I started thinking: *what if an agent — and an engineer training alongside it — could learn an optimal policy on a real cluster?*

---

## Enter KubeMedic

The name is intentional. *Kube* — because every fault, every broken pod, every evicted node is real Kubernetes. *Medic* — because the job isn't to observe the patient, it's to fix them. But underneath, the whole system runs on RL mechanics: observe state, take action, receive reward, update policy. KubeMedic is what happens when you stop treating on-call as a burden and start treating it as a **training environment**.

**The core design principle: stop giving the agent a fake environment.**

KubeMedic is an RL environment built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — Meta's open protocol for standardizing how agents interact with the world. The OpenEnv contract is clean: expose `/reset`, `/step`, `/state`, and `/ws` over HTTP+WebSocket, and any RL trainer — TorchRL, a PPO loop, a hand-rolled async rollout collector — can plug in without bespoke glue. Think of it as the successor to OpenAI Gym, but built for agents whose **action space** has real side effects — not grid moves, not Atari buttons, but live `kubectl` calls against a production-grade cluster.

Most RL environments abuse that simplicity by keeping the **environment dynamics** fake. KubeMedic takes the opposite trade: every **episode** runs against a real Azure Kubernetes Service cluster. Real pods. Real control plane. Real state transitions driven by the same Kubernetes reconcilers running in production.

```
Agent (LLM policy + curriculum controller)
        │  HTTP /reset, /step · WebSocket /ws  [OpenEnv protocol]
FastAPI Environment Server
        │
KubeMedicEnv  ←  episode state · trajectory buffer · reward engine
        │  kubernetes-python client
Real AKS Cluster (Azure)
```

Every `reset()` — the RL term for starting a new episode — tears down and recreates a challenge namespace, applies a base workload mesh (`api-gw`, `auth-svc`, `order-svc`, `payment-svc`), then injects exactly one real fault. The **initial state** `s₀` is what you get back: live pod and node observations pulled fresh from the API. The 15–30s reset cost is real and intentional. You're not loading a saved game state. You're waiting for Kubernetes reconcilers to propagate the injected fault into a stable broken state.

### The State Space — What the Agent Actually Sees

At each **timestep** `t`, the agent receives an **observation** — a partial view of the true environment state:

```python
class KubeMedicObservation(Observation):
    t: int                                # timestep within episode
    scenario: str | None                  # fault type (hidden at inference)
    pods: list[PodObservation]            # phase, conditions, restarts
    nodes: list[NodeObservation]          # pressure conditions, taints
    tool_result: dict[str, Any] | None    # result of last action
    blocked_reason: str | None            # guard rejection, if any
    info: dict[str, Any]                  # reward breakdown, diagnostics
```

This is a **partially observable MDP** — the agent never sees the ground truth fault directly. It has to infer the **latent state** (what's actually broken and why) from the observable signals (pod phases, event logs, resource metrics). That's exactly the epistemic situation a real SRE is in. You don't get a tooltip that says "memory limit too low." You get `OOMKilled` and you reason from there.

There's a debugging story buried in `tool_result` worth telling. In the first version, the per-tool payload — log lines, describe events, CPU millicores — was tucked inside `metadata` on the base `Observation` class. Then I noticed the agent's **policy** wasn't improving. Every tool call returned the same **observation vector** to the agent. `kubectl_logs`, `kubectl_describe`, `kubectl_top_pods` — indistinguishable. The policy gradient had nothing to differentiate on.

The cause: OpenEnv's `serialize_observation` does `model_dump(exclude={"reward", "done", "metadata"})`. The entire tool payload was being silently stripped on the wire. The fix was promoting `tool_result` to a declared Pydantic field so it survives serialization.

**Lesson: when your policy stops improving, suspect the observation channel before the learning algorithm.** Bad state representation kills gradient signal faster than a bad reward function.

### The Action Space — Same Tools as the Human

The agent's **action space** has 11 discrete tools. No oracle. No magic fix. Just what a real SRE reaches for:

**Read-only actions (zero side effects — safe to explore):**
`kubectl_get` · `kubectl_describe` · `kubectl_logs` · `kubectl_top_pods` · `kubectl_top_nodes`

**Mutating actions (irreversible within an episode — high consequence):**
`kubectl_patch_resources` · `kubectl_patch_tolerations` · `kubectl_delete_pod` · `kubectl_delete_workload` · `kubectl_cordon` · `kubectl_uncordon`

This asymmetry matters for **exploration strategy**. Read-only actions are safe to try; mutating ones carry real **transition risk** — a bad patch leaves the cluster in a worse state, a premature delete loses the diagnostic evidence you needed. A well-trained policy learns to **exploit** the read-only tools first to reduce uncertainty before committing to a mutating action.

Every mutating action is also gated by a guard layer. Writes into `kube-system`, force-deletes outside the challenge namespace, and scaling past 5 replicas are all blocked. Blocked actions return −5 reward and a `blocked_reason` the policy can condition on — mirroring the blast-radius constraints any real team encodes in their runbooks.

### The Fault Catalog — The MDP's Transition Dynamics

Five **environment scenarios**, each defining a distinct **transition function** between states:

| Scenario | Observable symptom | Latent root cause | Optimal action sequence |
|---|---|---|---|
| KUBE-01 | Node `MemoryPressure`, pods `Pending` | Stress pod requesting 3072Mi consuming the node | Delete hog → reap stuck pods |
| KUBE-03 | `payment-svc` `CrashLoopBackOff`, `OOMKilled` | Memory limit 256Mi, allocating 380Mi | Describe → logs-previous → patch limit ≥384Mi |
| KUBE-04 | `ml-inference` `Unschedulable` | Requests 14Gi + 6 cores — exceeds all node capacity | Describe → patch resource requests down |
| KUBE-05 | `gpu-workload` stuck `Pending` | Node tainted `gpu=true:NoSchedule`, no toleration in spec | Describe → patch tolerations |
| KUBE-06 | `DiskPressure`, pods evicted | Log-flood DaemonSet writing ~1Gi/min into `emptyDir` | Get nodes → delete DaemonSet → reap evicted pods |

These are not YAML flags. The stress containers use `polinux/stress` and genuinely consume memory. The taint lands on a real node. The DaemonSet fills disk at real I/O rates. Each fault defines a **stochastic environment** where the symptoms evolve over real time — an eviction that hasn't propagated yet, a pod that's mid-restart — so the **transition dynamics** are never perfectly deterministic from the agent's view.

### The Reward Function — Dense Signal Over Sparse Outcomes

This is the part I'm most deliberate about. Most RL environments use **sparse rewards** — +1 if you win, 0 otherwise. That's fine for games with millions of episodes. It's catastrophic for an environment where each episode costs 30 seconds on a cloud cluster and the **sample efficiency** budget is small.

KubeMedic uses a **dense, shaped reward** designed to give the policy gradient signal at every meaningful decision point, not just at the terminal state:

| Signal | Value | Policy behavior it reinforces |
|---|---|---|
| `STEP_PENALTY` | −0.25 | Discourages **episode length** bloat |
| `EARLY_PODS_SNAPSHOT` | +1.0 | Rewards initial **state observation** |
| `DESCRIBE_BROKEN_POD` | +2.0 | Rewards **targeted diagnosis** (not just any describe) |
| `LOGS_PREVIOUS` | +2.0 | Rewards accessing **pre-crash trajectory** |
| `TOP_PODS_REWARD` | +2.0 | Rewards **quantitative state estimation** before patching |
| `PREMATURE_MUTATION_PENALTY` | −2.0 | Punishes **acting before observing** |
| `SYMPTOM_ONLY_PENALTY` | −5.0 | Punishes **treating symptoms**, not root cause |
| `DISRUPTED_POD_PENALTY` | −25.0 | Punishes **negative side effects** on healthy pods |
| `CORRECT_FIX_BONUS` | +5.0 | Rewards **root-cause resolution** |
| `ALL_HEALTHY_TERMINAL` | +50.0 | **Terminal reward** for full recovery |
| `ZERO_DISRUPTIONS` | +25.0 | **Safety constraint** bonus |
| `SPEED_UNDER_5_STEPS` | +25.0 | **Efficiency** bonus |

A surgical 3-step fix with zero collateral damage scores ~+200. The same outcome achieved sloppily over 8 steps with one disrupted pod drops to ~+100. The full **reward decomposition** is returned in `info["reward_breakdown"]` so any critic model or human reviewer can trace exactly which components fired on which **timestep**.

The `PREMATURE_MUTATION_PENALTY` and `SYMPTOM_ONLY_PENALTY` are specifically designed to shape the **policy toward diagnosis-first behavior** — the same behavior instilled in SREs by good incident culture. Without them, the policy quickly finds a shortcut: spam mutating actions until the cluster recovers. With them, the optimal **trajectory** requires evidence-gathering before action, because premature mutation bleeds reward every time.

### The Curriculum — Non-Stationary Environment Distribution

The `CurriculumController` is what separates KubeMedic from a static benchmark. In standard RL, the environment distribution is fixed. Here, it deliberately shifts as the **agent's skill level** improves — a form of **curriculum learning** where the task difficulty tracks competency.

Five tiers: `warmup → beginner → intermediate → advanced → expert`. Each tier gates which fault types are reachable. `oom_kill` and `memory_pressure` are available from day one. `disk_pressure` only unlocks at the intermediate threshold. The agent has to **graduate** the easier scenarios before harder ones enter the **episode distribution**.

`pick_scenario()` implements a **non-uniform sampling policy** over the available fault types:
- **Untried faults first** — pure **exploration** of unseen states
- **Weak spots next** — fault types with mastery rate < 0.7 on the rolling 10-episode window
- **3:1 weighting** — non-graduated faults over graduated ones, keeping the **exploitation** pressure on improvement areas

A fault graduates when its rolling 10-episode success rate hits 70%. Fast-track triggers at ≥3 consecutive episodes above 90% — the curriculum recognizes when the **policy has converged** on a fault type and stops wasting sample budget there.

The LLM judge scoring each **episode transcript** also scales its persona with tier. At warmup it's a *"junior SRE mentor"* — lenient on efficiency, checking for basic diagnostic intent. At expert it's a *"principal SRE reviewer"* — strict on blast radius, **trajectory quality**, and whether the fix was the minimal correct intervention. The same agent trajectory that scores 8/10 from a junior judge might score 5/10 from a principal. **The agent only advances when it can satisfy the harder evaluator** — a key design choice that prevents the policy from overfitting to a lenient reward signal.

> *"Build something 100 people love, not something 1 million people kind of like."* — Garry Tan. KubeMedic isn't trying to be a general observability platform. It's trying to build policies — and engineers — that are genuinely great at one thing: recovering a broken cluster without making it worse.

In a 12-episode curriculum run, the agent moved warmup → expert, graduating `oom_kill` and `taint_mismatch` along the way, with **cumulative return** stabilizing around 169 per episode. Not a solved MDP — but a clear signal that dense reward shaping and curriculum distribution shift compose the way they should.

---

## A Trajectory Walkthrough — KUBE-03

Let me trace a full **agent trajectory** through KUBE-03 — the scenario closest to my 3 AM.

**t=0 (reset):** `payment-svc` is in `CrashLoopBackOff`. Initial observation shows pod phase, restart count, no tool result. The **policy** has to decide its first action with only pod-level state.

**t=1 (action: `kubectl_describe`, payment-svc):** `tool_result` returns `OOMKilled` in container status, memory limit 256Mi. **+2.0 reward** — the policy reached for diagnosis on the right pod. **Latent state partially revealed.**

**t=2 (action: `kubectl_logs`, previous=True):** Log lines show the stress process allocating 380Mi before the OOM kill. **+2.0 reward** — pre-crash **trajectory** accessed. The policy now has enough evidence to localize the root cause: limit is wrong, not the pod.

**t=3 (action: `kubectl_patch_resources`, limit=400Mi):** Strategic merge patch applied. Deployment reconciler restarts the container. **+5.0 correct fix bonus** fires. Pod transitions to Running.

**t=4 (action: `kubectl_get`, pods):** Confirmation step. All pods Running. **+1.0 post-fix confirmation**. Episode **terminates** — `done=True`.

**Terminal rewards: +50 (all healthy) + +25 (zero disruptions) + +25 (under 5 steps).** Total episode return: ~+205.

The lazy **policy** — delete the pod, hope it restarts — would miss the PREMATURE_MUTATION_PENALTY and the SYMPTOM_ONLY_PENALTY, fail to recover (the spec is unchanged), and exhaust `max_steps` with a sparse terminal reward of 0. The shaped reward makes the difference **learnable from a small number of episodes**.

---

## Impact and What I See Ahead

Every second a customer-facing service is dark, you're spending trust you might not have earned back yet. KubeMedic is an attempt to answer a real question: *can we train policies with the same calibrated caution a good senior SRE has?* Not just knowing what `kubectl describe` does — knowing when the **expected value** of gathering more evidence exceeds the **opportunity cost** of delay.

Five fault scenarios right now. Real production has hundreds — network policy failures, image pull auth errors, PVC binding bugs, admission webhook rejections. Each is a new **MDP** with different **transition dynamics** and a different **optimal policy**. The next step I'm most excited about is **adversarial scenario generation** — a separate model acting as an environment generator, proposing new fault recipes once the agent graduates to expert tier, ensuring the **episode distribution** keeps shifting past the hand-authored ceiling.

The bigger vision: a training substrate where a new SRE engineer and an LLM agent run the same **rollouts** side by side, under a judge that won't let either hand-wave the diagnostic process. A shared **graduation bar** where advancing to the next tier means something real, because the **value function** of experience has been honestly earned.

We spend enormous resources building observability tooling. We spend almost nothing building the **training environments** where we teach people — and now policies — to use it under pressure.

That's the gap KubeMedic is trying to close.

The portal came back up at 4:19 AM. I closed the laptop and stared at the ceiling. Three weeks later I was 27 hours into a 30-hour hackathon, running an agent through its first real episode on a live AKS cluster, watching it describe a pod before it patched anything — and thinking: yeah, that's the one.

Sometimes the most useful thing you can build isn't better alerting. It's a better **reward signal**.

---

*Built for the Meta Scaler Hackathon · OpenEnv-compatible · Trains against real Azure Kubernetes Service clusters*
