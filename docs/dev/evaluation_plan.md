# Evaluation Implementation Plan — Accepted Standard Protocol

## 1. What the Standard Protocol Is

After thorough review of the literature (LIBERO NeurIPS 2023, OpenVLA, Pi0.5, GR00T N1.6, LeRobot/HuggingFace, LIBERO-PRO), the accepted evaluation method for LIBERO is:

| Element | Standard | Source |
|---------|----------|--------|
| **Evaluation type** | Environment-based rollout in robosuite (MuJoCo) | LIBERO paper §5, OpenVLA, Pi0.5, GR00T |
| **Metric** | Binary task success via goal predicates | LIBERO Appendix E.2: "success rates, not BC loss" |
| **Episodes per task** | **50** (standard) or **10** (minimum) | OpenVLA: 50; LeRobot: 10 |
| **Total trials per suite** | 500 (10 tasks × 50 episodes) or 400 (4 suites × 10 tasks × 10 episodes) | OpenVLA §4, LeRobot docs |
| **Suites** | LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-Long (LIBERO-10) | LIBERO §4.2 |
| **Seeds** | 3 random seeds, report mean ± standard error | OpenVLA, all SOTA work |
| **Initial states** | Sampled i.i.d. from training distribution (different from training states) | OpenVLA issue #138, Kim et al. |
| **Action space** | 7-DoF continuous: 6D delta EE pose + 1D gripper | LIBERO §4 |
| **Observation** | RGB images (agentview + wrist) + 8-DoF proprioception | Standard VLA setting |
| **Max steps** | Suite-specific: Spatial=280, Object=280, Goal=300, Long=520 | LeRobot `TASK_SUITE_MAX_STEPS` |

### 1.1 What This Means for PhaseForge

PhaseForge is a **state-only** policy (23-DoF state input, no images). No published SOTA exists for state-only LIBERO — the community uses vision. This means:

1. Our evaluation protocol will be **novel** — we define the state-only standard
2. We cannot compare success rates directly to vision-based SOTA (97–98%)
3. We should evaluate on **LIBERO-90** (our training tasks) and optionally **LIBERO-LONG** (transfer)
4. We must report **per-suite breakdowns** (not just a single aggregate)
5. We should run **3 seeds** for statistical significance

### 1.2 Warning: The Standard Has Known Limitations

LIBERO-PRO (Zhou et al., 2025) demonstrates that even the standard rollout protocol overestimates capability: models at >90% on standard LIBERO collapse to <30% under modest perturbations (object displacement, instruction rewriting). The standard protocol tests near-identical train/test distributions.

**Implication:** Our state-only results on the standard protocol may also overestimate true generalization. We should note this as a limitation and potentially adopt LIBERO-PRO-style perturbations in a follow-up.

---

## 2. Implementation Plan

### 2.1 Dependencies to Add

| Dependency | Version | Purpose | Installation |
|-----------|---------|---------|-------------|
| `robosuite` | ==1.4.0 | Franka Panda simulation | `uv sync --extra rollout` |
| `mujoco` | ==3.3.1 | Physics engine (robosuite dep) | installed with robosuite |
| `gymnasium` | 1.3.0 (resolved via robosuite, uv.lock) | Environment interface | installed with robosuite |
| `libero` | ==0.1.1 (resolved in uv.lock) | Task definitions, initial states, benchmark | `uv add libero` |

The official LIBERO package provides:
- `libero.benchmark.get_benchmark_dict()` — loads task suites
- `libero.benchmark.TaskSuite` — iterable of tasks with BDDL files
- Initial state files for each task
- Environment creation helpers

**Installation order:**
```
uv sync --extra rollout  # robosuite==1.4.0, mujoco==3.3.1, libero==0.1.1 (uv.lock)
```

Note: robosuite and MuJoCo require a display server (X11 on Linux, or use EGL/OSMesa). On headless servers, use `xvfb` or `egl` rendering.

### 2.2 Exact LIBERO APIs (from Official Codebase)

The official LIBERO benchmark provides these exact APIs (verified from `libero` PyPI package v0.1.1 and HuggingFace `lerobot/lerobot`):

```python
# --- BENCHMARK LOADING ---
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = benchmark.get_benchmark_dict()
# Available keys: "libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10", "libero_100"
task_suite = benchmark_dict["libero_spatial"]()
num_tasks = task_suite.n_tasks                         # always 10 for Spatial/Object/Goal/Long
task = task_suite.get_task(task_id)                     # task_id in 0..9
task_name = task.name                                   # e.g. "LIVING_ROOM_SCENE1_pick_up_the_black_bowl"
task_description = task.language                        # e.g. "pick up the black bowl between the plate..."

task_bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file,
)

# --- ENVIRONMENT CREATION ---
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,   # must be set but we don't use images
    "camera_widths": 128,
}
env = OffScreenRenderEnv(**env_args)

# --- ENVIRONMENT RESET SEQUENCE (CRITICAL - must follow this exact order) ---
env.seed(seed)                                          # affects object positions even with fixed init states
env.reset()                                             # must come before set_init_state
init_states = task_suite.get_task_init_states(task_id)  # shape: (50, 123) = 50 episodes × 123 sim state values
obs = env.set_init_state(init_states[episode_idx])      # returns observation dict

# --- NUM WAIT STEPS ---
num_steps_wait = 10
dummy_action = [0.0] * 7
for _ in range(num_steps_wait):
    obs, reward, done, info = env.step(dummy_action)    # let objects settle

# --- OBSERVATION DICT KEYS (from robosuite) ---
obs["robot0_joint_pos"]          # (7,)  - joint positions
obs["robot0_joint_vel"]          # (7,)  - joint velocities
obs["robot0_eef_pos"]            # (3,)  - end effector position
obs["robot0_eef_quat"]           # (4,)  - end effector quaternion
obs["robot0_gripper_qpos"]       # (2,)  - gripper finger positions
obs["robot0_proprio-state"]      # (8,)  - concatenated: eef_pose(6 via axis-angle) + gripper_qpos(2)
obs["agentview_image"]           # (128, 128, 3) - third-person camera (we ignore)
obs["robot0_eye_in_hand_image"]  # (128, 128, 3) - wrist camera (we ignore)

# --- STATE VECTOR CONSTRUCTION (23-DoF for our model) ---
state_23dof = np.concatenate([
    obs["robot0_joint_pos"],      # 7
    obs["robot0_joint_vel"],      # 7
    obs["robot0_eef_pos"],        # 3
    obs["robot0_eef_quat"],       # 4
    obs["robot0_gripper_qpos"],   # 2
]).astype(np.float32)             # total: 23

# --- ACTION ---
action = np.array([0.0] * 7)     # 7-DoF: 6D delta EE pose + 1D gripper
obs, reward, done, info = env.step(action)

# --- SUCCESS CHECK ---
is_success = env.check_success()                          # method 1
# OR
is_success = info.get("is_success", False)                # method 2 (some versions)

# --- TERMINATION ---
# done = True when max episode steps reached
# robosuite raises ValueError: "executing action in terminated episode" if you step past max_steps
# Always check done before stepping

# --- MAX STEPS PER SUITE ---
MAX_STEPS = {
    "libero_spatial": 220,   # longest training demo: 193 steps
    "libero_object": 280,    # longest training demo: 254 steps
    "libero_goal": 300,      # longest training demo: 270 steps
    "libero_10": 520,        # longest training demo: 505 steps
    "libero_90": 400,        # longest training demo: 373 steps
}

# --- ENVIRONMENT CLOSE ---
env.close()
```

### 2.3 Architecture: New Files to Create

```
phaseforge/evaluations/
├── __init__.py
├── runners/
│   ├── __init__.py
│   ├── offline_evaluator.py        # unchanged
│   └── rollout_evaluator.py        # REWRITE — environment-based eval
├── envs/
│   ├── __init__.py
│   └── libero_env.py               # NEW — state-only LIBERO env wrapper
└── metrics/
    ├── __init__.py                 # unchanged
    ├── task_metrics.py             # unchanged
    ├── routing_stability.py        # unchanged
    ├── phase_alignment.py          # unchanged
    └── expert_utilization.py       # unchanged
```

### 2.4 File 1: `envs/libero_env.py` — State-Only LIBERO Environment Wrapper

**Purpose:** Wraps `OffScreenRenderEnv` to produce only the 23-DoF state vector our models expect, discarding images. Also loads initial states from the benchmark, handles the required env reset sequence, and provides `check_success()`.

```python
import os
import numpy as np
from typing import Any

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

SUITE_BENCHMARK_NAMES = {
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
    "libero_long": "libero_10",
    "libero_10": "libero_10",
    "libero_90": "libero_90",
}


class StateOnlyLiberoEnv:
    """
    State-only wrapper around LIBERO's OffScreenRenderEnv.

    Constructs the 23-DoF state vector (joint_pos 7 + joint_vel 7 + eef_pos 3
    + eef_quat 4 + gripper_qpos 2) that matches our training data format.
    All image observations are discarded.

    Usage:
        env = StateOnlyLiberoEnv(suite_name="libero_spatial", task_id=0, seed=42)
        state = env.reset(episode_idx=0)       # 23-DoF numpy array
        next_state, reward, done, info = env.step(action)  # action is (7,) numpy
        env.close()
    """

    def __init__(
        self,
        suite_name: str,
        task_id: int,
        seed: int = 42,
        num_steps_wait: int = 10,
        camera_heights: int = 128,
        camera_widths: int = 128,
    ):
        bench_name = SUITE_BENCHMARK_NAMES[suite_name]
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[bench_name]()
        self.task_id = task_id
        self.suite_name = suite_name
        self.seed = seed
        self.num_steps_wait = num_steps_wait
        self.max_steps = SUITE_MAX_STEPS[suite_name]
        self._elapsed_steps = 0
        self._task = self.task_suite.get_task(task_id)
        self._init_states = self.task_suite.get_task_init_states(task_id)
        self._env = None

        bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            self._task.problem_folder,
            self._task.bddl_file,
        )
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
        )

    @property
    def task_description(self) -> str:
        return self._task.language

    @property
    def num_init_states(self) -> int:
        return len(self._init_states)

    def _extract_state(self, obs: dict) -> np.ndarray:
        return np.concatenate([
            obs["robot0_joint_pos"],
            obs["robot0_joint_vel"],
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
            obs["robot0_gripper_qpos"],
        ]).astype(np.float32)

    def reset(self, episode_idx: int = 0) -> np.ndarray:
        if episode_idx >= len(self._init_states):
            raise IndexError(
                f"episode_idx={episode_idx} out of range "
                f"(available init states={len(self._init_states)})"
            )
        self._env.seed(self.seed)
        self._env.reset()
        obs = self._env.set_init_state(self._init_states[episode_idx])
        dummy = np.array([0.0] * 7, dtype=np.float32)
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self._env.step(dummy)
        self._elapsed_steps = 0
        return self._extract_state(obs)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}")
        obs, reward, done, info = self._env.step(action)
        self._elapsed_steps += 1
        is_success = self._env.check_success()
        terminated = is_success or done
        truncated = (not terminated) and (self._elapsed_steps >= self.max_steps)
        state = self._extract_state(obs)
        info_out = {
            "is_success": bool(is_success),
            "elapsed_steps": self._elapsed_steps,
        }
        return state, float(reward), terminated, truncated, info_out

    def close(self):
        if self._env is not None:
            self._env.close()
```

### 2.5 File 2: `runners/rollout_evaluator.py` — Rollout Evaluation Engine

**Purpose:** Orchestrates the full evaluation across all tasks in a suite, running per-episode rollouts. Collects success rates, MoE routing metrics, and per-task breakdowns. Follows the OpenVLA evaluation pattern exactly.

```python
import torch
import numpy as np
from pathlib import Path
from typing import Any
from omegaconf import DictConfig

from phaseforge.evaluations.envs.libero_env import StateOnlyLiberoEnv, SUITE_MAX_STEPS
from phaseforge.utils.registry import build_data_pipeline


class RolloutEvaluator:
    """
    Environment-based rollout evaluator for LIBERO.

    Runs the standard evaluation protocol:
    - For each task in each configured suite
    - For each episode (num_episodes_per_task)
    - Reset env to initial state, run policy until done/max_steps
    - Record binary success from env.check_success()
    - Aggregate per-suite and overall success rates
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        device: torch.device,
    ):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.model.eval()

        # Load normalizer from training data pipeline
        pipeline = build_data_pipeline(cfg)
        dataset = pipeline._datasets["val"] if hasattr(pipeline, "_datasets") else None
        if dataset is not None and hasattr(dataset, "normalizer"):
            self.normalizer = dataset.normalizer
        else:
            self.normalizer = None

    @torch.no_grad()
    def _get_action(self, state: np.ndarray) -> np.ndarray:
        """Run model inference: state (23,) -> action (7,)."""
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        # Apply normalization if available
        if self.normalizer is not None:
            state_tensor = self.normalizer.normalize_state(state_tensor)

        # Create batch dict as expected by model.forward
        batch = {"state": state_tensor}

        # Use get_action for inference (no aux outputs, no gradients)
        action = self.model.get_action(state_tensor)  # (1, 7) or (7,)

        if action.ndim == 2:
            action = action.squeeze(0)

        # Un-normalize action if normalizer exists
        if self.normalizer is not None:
            action_np = self.normalizer.unnormalize_action(action.cpu().numpy())
        else:
            action_np = action.cpu().numpy()

        return action_np.astype(np.float64)

    def evaluate_suite(
        self,
        suite_name: str,
        num_episodes_per_task: int,
    ) -> dict[str, Any]:
        """Evaluate a single LIBERO suite. Returns per-task and aggregate results."""
        bench_key = {
            "libero_spatial": "libero_spatial",
            "libero_object": "libero_object",
            "libero_goal": "libero_goal",
            "libero_long": "libero_10",
            "libero_90": "libero_90",
        }[suite_name]

        # Get number of tasks in suite
        from libero.libero import benchmark
        bd = benchmark.get_benchmark_dict()
        task_suite = bd[bench_key]()
        num_tasks = task_suite.n_tasks

        all_routing_weights = []
        all_expert_indices = []
        all_gate_logits = []
        all_phases = []

        total_episodes = 0
        total_successes = 0
        per_task_results: dict[str, dict] = {}

        for task_id in range(num_tasks):
            env = StateOnlyLiberoEnv(
                suite_name=suite_name,
                task_id=task_id,
                seed=self.cfg.project.seed,
                num_steps_wait=self.cfg.eval.environment.get("num_steps_wait", 10),
            )

            task_desc = env.task_description
            max_steps = SUITE_MAX_STEPS[suite_name]
            task_successes = 0

            for episode_idx in range(num_episodes_per_task):
                state = env.reset(episode_idx=episode_idx)
                episode_success = False

                for step in range(max_steps):
                    action = self._get_action(state)
                    state, reward, terminated, truncated, info = env.step(action)

                    if terminated or truncated:
                        if info.get("is_success", False):
                            episode_success = True
                            task_successes += 1
                            total_successes += 1
                        break

                total_episodes += 1

            env.close()

            task_success_rate = task_successes / num_episodes_per_task
            per_task_results[task_desc] = {
                "success_rate": task_success_rate,
                "successes": task_successes,
                "episodes": num_episodes_per_task,
            }

        overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0

        return {
            f"eval/success_rate/{suite_name}": overall_success_rate,
            f"eval/per_task/{suite_name}": per_task_results,
            f"eval/total_episodes/{suite_name}": total_episodes,
            f"eval/total_successes/{suite_name}": total_successes,
        }

    def run(self) -> dict[str, Any]:
        # Decision 2 (issues register A2, 2026-08-07): libero_90 is the
        # in-distribution core; libero_10 is the labeled zero-shot row.
        # Spatial/object/goal are NOT fallback defaults.
        suites = self.cfg.eval.environment.get("suites", ["libero_90"])
        num_ep = self.cfg.eval.evaluation.get("num_episodes_per_task", 50)

        all_results: dict[str, Any] = {}
        all_suite_rates = []

        for suite_name in suites:
            suite_results = self.evaluate_suite(suite_name, num_episodes_per_task=num_ep)
            all_results.update(suite_results)
            rate_key = f"eval/success_rate/{suite_name}"
            if rate_key in suite_results:
                all_suite_rates.append(suite_results[rate_key])

        # Aggregate across all suites
        if all_suite_rates:
            all_results["eval/success_rate"] = float(np.mean(all_suite_rates))

        all_results["eval/seed"] = self.cfg.project.seed
        all_results["eval/num_episodes_per_task"] = num_ep
        all_results["eval/suites"] = list(suites)

        return all_results
```

### 2.6 File 3: `cli.py` — Evaluation Command Integration

**Extend `evaluate()` function** to support both offline and rollout modes:

```python
@hydra.main(version_base="1.3", config_path="config", config_name="main")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model on the validation/test set."""
    set_seed(cfg.project.seed)

    # Determine eval mode from config
    eval_mode = cfg.eval.get("mode", "rollout")

    if eval_mode == "offline":
        # --- existing offline evaluation path ---
        output_dir = get_eval_output_dir(cfg)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "resolved_config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        write_run_meta(output_dir, cfg)
        logger.info(f"Evaluation output directory: {output_dir}")

        pipeline = build_data_pipeline(cfg)
        dataloaders = pipeline.run()
        val_loader = dataloaders.get("val") or dataloaders.get("test")
        if val_loader is None:
            raise RuntimeError("No validation/test data found for evaluation.")

        model = build_model(cfg)
        ckpt_path = cfg.train.get("stage1_ckpt_path")
        if ckpt_path:
            logger.info(f"Loading checkpoint from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if hasattr(model, "stage") and "stage" in ckpt:
                model.stage = ckpt["stage"]
        else:
            logger.warning("No checkpoint provided. Using randomly initialized model.")

        model.to(cfg.project.get("device", "cuda"))
        evaluator = OfflineEvaluator(cfg=cfg, model=model, dataloader=val_loader)
        results = evaluator.run()

        results_path = output_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    elif eval_mode == "rollout":
        # --- new rollout evaluation path ---
        output_dir = get_eval_output_dir(cfg)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = build_model(cfg)
        ckpt_path = cfg.train.get("stage1_ckpt_path")
        if ckpt_path:
            logger.info(f"Loading checkpoint from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if hasattr(model, "stage") and "stage" in ckpt:
                model.stage = ckpt["stage"]
        else:
            logger.warning("No checkpoint provided. Using randomly initialized model.")

        device = torch.device(cfg.project.get("device", "cuda"))
        model.to(device)

        evaluator = RolloutEvaluator(cfg=cfg, model=model, device=device)
        results = evaluator.run()

        results_path = output_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    else:
        raise ValueError(f"Unknown eval mode: {eval_mode}. Use 'offline' or 'rollout'.")

    logger.info("Evaluation complete:")
    for key, val in results.items():
        if isinstance(val, float):
            logger.info(f"  {key}: {val:.6f}")
        elif isinstance(val, dict):
            logger.info(f"  {key}: {json.dumps(val, indent=4)}")
```

### 2.7 Config Changes

**`config/eval/rollout.yaml`** — replace placeholder with real config:

```yaml
# config/eval/rollout.yaml
mode: "rollout"

environment:
  # Decision 2 (issues register A2, 2026-08-07): libero_90 (ID) +
  # libero_10 (labeled zero-shot row) ONLY. Spatial/object/goal are
  # separate transfer-study benchmarks, not eval suites for a
  # libero_90-trained agent.
  suites: ["libero_90", "libero_10"]
  control_mode: "relative"
  episode_length: null                # use suite defaults (libero_10=520, libero_90=400)
  num_steps_wait: 10                  # wait steps for sim objects to settle

evaluation:
  num_episodes_per_task: 50           # standard: 50, minimum: 10 (LeRobot)
  episodes_per_suite: {libero_90: 50, libero_10: 10}  # E5: 10 eps on the zero-shot row
  seeds: [42, 43, 44]                 # 3 seeds for statistical significance
  record_video: false
  video_dir: null
```

### 2.8 Multi-Seed Run Script

**`scripts/run_multi_seed_eval.py`** — convenience script for 3-seed evaluation:

```python
"""Run 3-seed rollout evaluation for a single model checkpoint."""
import subprocess
import json
from pathlib import Path

MODELS = [
    ("phaseforge", "outputs/phaseforge/stage2/2026-07-17_09-27-55_f63dfb08/checkpoints/checkpoint_best.pt"),
    ("bc", "outputs/bc/stage1/2026-07-17_09-50-03_29c621df/checkpoints/checkpoint_best.pt"),
    ("scratch_moe", "outputs/scratch_moe/stage2/2026-07-17_09-56-59_0d627795/checkpoints/checkpoint_best.pt"),
    ("warmstart_moe", "outputs/warmstart_moe/stage2/2026-07-17_10-13-36_29b20708/checkpoints/checkpoint_best.pt"),
    ("oracle_moe", "outputs/oracle_moe/stage2/2026-07-17_10-21-41_aa28a106/checkpoints/checkpoint_best.pt"),
]

SEEDS = [42, 43, 44]

def run_eval(model_name, ckpt_path, seed):
    cmd = [
        "uv", "run", "phaseforge-eval",
        f"models={model_name}",
        f"train.stage1_ckpt_path={ckpt_path}",
        "eval=rollout",
        f"project.seed={seed}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)

if __name__ == "__main__":
    all_results = {}
    for model_name, ckpt_path in MODELS:
        per_seed = []
        for seed in SEEDS:
            result = run_eval(model_name, ckpt_path, seed)
            per_seed.append(result)
        # Aggregate mean ± std across seeds
        success_rates = [r["eval/success_rate"] for r in per_seed]
        all_results[model_name] = {
            "mean": np.mean(success_rates),
            "std": np.std(success_rates),
            "per_seed": per_seed,
        }
        print(f"{model_name}: {all_results[model_name]['mean']:.4f} ± {all_results[model_name]['std']:.4f}")

    with Path("outputs/eval/final_results.json").open("w") as f:
        json.dump(all_results, f, indent=2)
```

### 2.8 Reporting Format

Following the published standard, the final table should look like:

| Model | Spatial | Object | Goal | Long | Average |
|-------|:-------:|:------:|:----:|:----:|:-------:|
| BC | XX.X ± X.X | XX.X ± X.X | XX.X ± X.X | XX.X ± X.X | XX.X |
| PhaseForge | XX.X ± X.X | XX.X ± X.X | XX.X ± X.X | XX.X ± X.X | XX.X |
| Scratch MoE | XX.X ± X.X | XX.X ± X.X | XX.X ± X.X | XX.X ± X.X | XX.X |
| WarmStart MoE | XX.X ± X.X | XX.X ± X.X | XX.X ± X.X | XX.X ± X.X | XX.X |
| Oracle MoE | XX.X ± X.X | XX.X ± X.X | XX.X ± X.X | XX.X ± X.X | XX.X |

Each cell = mean ± standard error over 3 seeds.

---

## 3. Implementation Order

| Step | What | Dependencies | Expected Effort |
|------|------|-------------|-----------------|
| 1 | Install robosuite + libero | None | 1 hour |
| 2 | Write `envs/libero_env.py` — state-only wrapper | Step 1 | 2 days |
| 3 | Write `runners/rollout_evaluator.py` — rollout loop | Step 2 | 3 days |
| 4 | Write `cli.py` eval mode switch | Step 3 | 2 hours |
| 5 | Write `config/eval/rollout.yaml` | Step 4 | 1 hour |
| 6 | Test: single task, single episode, verify state normalization | Step 5 | 2 days |
| 7 | Test: full suite (Spatial/10 tasks × 50 episodes) | Step 6 | 1 day |
| 8 | Write multi-seed script + run all 5 models × 3 seeds | Step 7 | 1 day |
| 9 | Write `scripts/run_multi_seed_eval.py` | Step 8 | 2 hours |
| **Total** | | | **~10 working days** |

### 3.1 Most Important Test (Step 6)

Before running the full evaluation, verify that the state vector produced by `StateOnlyLiberoEnv` matches the training-time state format:

1. Load a training HDF5 demo trajectory
2. Reset the env to the same initial state used in the demo
3. Step the env with the demo's actions
4. Compare the env's state output to the HDF5's recorded state at each timestep

If the state vectors match within simulation tolerance, the evaluation is valid. If they diverge (due to different sim versions, controller settings, etc.), the evaluation will not measure what we trained on.

---

## 4. Reporting Standard Compliance Checklist

| Requirement | Our Implementation | Status |
|-------------|-------------------|--------|
| Simulator rollouts (MuJoCo) | `StateOnlyLiberoEnv` → robosuite | ❌ Not implemented |
| Binary task success from environment | `info["is_success"]` | ❌ Not implemented |
| 50 episodes per task | `num_episodes_per_task=50` | ❌ Config not written |
| 3 seeds | `seeds=[42,43,44]` | ❌ Config not written |
| Per-suite breakdown | Spatial, Object, Goal, Long separately | ❌ Not implemented |
| Mean ± std across seeds | `eval/success_rate_std` | ❌ Script not written |
| Per-task success rates | `eval/per_task_success` dict | ❌ Not implemented |
| Max steps per suite | Spatial=280, Object=280, Goal=300, Long=520 | ❌ Not configured |
| Initial states different from training | Default LIBERO initial state loading | ❌ Not implemented |
| MoE routing metrics during rollouts | Gate logits + phase labels from env | ❌ Not implemented |
| Action normalization vs. raw actions | Must match training-time normalization | ❌ Not implemented |

---

## 5. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|-----------|
| robosuite + libero installation fails on headless server | Medium | High | Use `xvfb-run` or EGL rendering; test on local machine first |
| State vector mismatch between training (HDF5) and robosuite output | High | Critical | Step 6 test: replay demo actions, compare state trajectories |
| Normalization mismatch: rollout states unnormalized but model trained on normalized | Medium | Critical | Store `normalizer_mean.pt` / `normalizer_std.pt` during data pipeline; load during rollout |
| Action range mismatch: model outputs normalized actions, env expects unnormalized | Medium | Critical | Config must specify whether model outputs raw or normalized actions; add inverse transform |
| LIBERO suite definition for state-only differs from standard | Medium | Medium | The `libero` package's task definitions are vision-agnostic; state-only wrapper just selects which obs keys to return |
| Different MuJoCo/robosuite versions produce different physics | Low | Medium | Pin versions in `pyproject.toml`; document exact version numbers in paper |
| Rollout evaluation is compute-intensive (500 episodes × 4 suites × 3 seeds = 6000 episodes) | Low | Medium | Not a risk but a time budget estimate; ~3-5 days on a single GPU depending on model inference speed |

---

## 6. Acceptance Criteria

The evaluation pipeline is complete when:

1. [ ] `StateOnlyLiberoEnv.reset()` returns a 23-DoF state vector matching training format
2. [ ] `StateOnlyLiberoEnv.step(action)` returns (next_state, reward, terminated, info) where info["is_success"] is binary
3. [ ] State trajectory matches HDF5 replay within MuJoCo tolerance (Step 6 test)
4. [ ] Rollout evaluator runs all 10 tasks in LIBERO-Spatial at 50 episodes each = 500 episodes
5. [ ] Rollout evaluator produces per-task and aggregate success rates
6. [ ] Rollout evaluator captures MoE routing metrics (gate_logits, expert_indices) during rollouts
7. [ ] Multi-seed script runs all 5 models × 3 seeds unattended
8. [ ] Final results table shows per-suite breakdown with mean ± std across seeds

---

## 7. Key References

| Reference | Link | Relevance |
|-----------|------|-----------|
| LeRobot LIBERO docs (HuggingFace) | https://huggingface.co/docs/lerobot/en/libero | Standard protocol: 10 episodes/task, 4 suites |
| OpenVLA LIBERO eval script | https://raw.githubusercontent.com/moojink/openvla-oft/main/experiments/robot/libero/run_libero_eval.py | Reference implementation: 50 episodes/task, 500 trials per suite |
| LIBERO GitHub | https://github.com/Lifelong-Robot-Learning/LIBERO | Benchmark definitions, initial states |
| LIBERO-PRO (Zhou et al.) | https://arxiv.org/abs/2510.03827 | Demonstrates standard protocol overestimates capability |
| ROBOSUITE docs | https://robosuite.ai/docs/ | Simulation framework |
| PhaseForge config/common.yaml | `config/data/common.yaml` | State dimension order (23-DoF) |
| PhaseForge PhaseAwareCollator | `phaseforge/data/collator.py` | Training-time state format |
