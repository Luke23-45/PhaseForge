"""Rollout evaluator using LIBERO environment.

Runs the standard evaluation protocol:
- For each task in each configured suite
- For each episode, reset env to initial state, run policy until done/max_steps
- Record binary success from the EXPLICIT ``env.check_success()``
- Aggregate per-suite and overall success rates

Correctness contracts (issues register E9):
- Actions are validated (exact 7 dims, finite) before every env.step and
  NEVER silently clipped; the policy is recorded in ``eval/action_policy``.
- The env state vector is validated (shape == normalizer/model input dim,
  finite) before the first model call of each episode.
- Per-task results are keyed by numeric task ID (descriptions are retained
  as metadata); task counts from the installed benchmark are asserted
  against the protocol's suite definitions at startup.
- Episode counts / wait steps / worker counts / suite names are validated
  up front — a zero episode count can never reach an obscure shard error.
- The results payload records full provenance: checkpoint path + SHA-256
  + its run_meta, git commit, config/data/cache/object-index hashes, the
  effective worker count, per-suite episode counts, environment versions,
  and the installed controller identity (train/eval action semantics).

Performance notes (verified against the LIBERO/robosuite stack):
- Unused observables are pruned by default (``render_observations: false``):
  camera images AND object sensors are disabled, keeping only the five
  observables that build the 23-DoF state vector. Per-step rendering and
  sensor updates are the dominant per-step costs.
- Per-step cost is physics-bound (~25 MuJoCo substeps per control step,
  single-threaded per env), so wall-clock throughput scales with vCPUs.
  ``num_workers: 0`` (the default) auto-resolves to one worker per logical
  CPU; set an explicit value to cap it on small VRAM cards.
- With ``num_workers > 1`` episodes are sharded round-robin across spawned
  worker processes (the vla-eval/LIBERO-recommended pattern), so the GPU can
  serve action inference while other workers simulate. Results are
  bit-identical to the serial path because episodes are deterministic
  functions of ``init_states[episode_idx]``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from phaseforge.data.common.normalizer import FrozenNormalizer
from phaseforge.data.ingestion.cache_manager import CacheManager
from phaseforge.data.paths import processed_cache_root
from phaseforge.evaluations.envs.libero_env import (
    SUITE_BENCHMARK_NAMES,
    SUITE_MAX_STEPS,
    StateOnlyLiberoEnv,
)
from phaseforge.utils.config import git_info

logger = logging.getLogger(__name__)

# B4: declare the evaluation role of every suite in the results. Per
# Decision 2 (issues register A2) only libero_90 (in-distribution) and
# libero_10 (labeled zero-shot row) are evaluated; anything else is
# declared "unclassified" rather than silently implied to be ID.
SUITE_ID_OOD_ROLES: dict[str, str] = {
    "libero_90": "in-distribution",
    "libero_10": "zero-shot (labeled)",
}

#: Official task counts per LIBERO suite (E9): the runtime benchmark's
#: ``n_tasks`` is asserted against these before any rollout runs, so a
#: mismatched LIBERO version can never silently evaluate a different grid
#: than the one the summary code assumes (issues register E9, item 12).
SUITE_N_TASKS: dict[str, int] = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}

ACTION_DIM_DEFAULT = 7  # OSC_POSE delta pose + gripper (LIBERO convention)


def _resolve_num_workers(requested: int, num_episodes: int) -> int:
    """Resolve the effective worker count.

    ``requested <= 0`` means auto: one worker per logical CPU (physics is
    single-threaded per env, so throughput scales with vCPUs). Either way
    the count is capped so we never spin up more workers than episodes.
    """
    if int(requested) <= 0:
        requested = os.cpu_count() or 1
    return max(1, min(int(requested), num_episodes))


def split_episode_shards(num_episodes: int, num_workers: int) -> list[list[int]]:
    """Split absolute episode indices across workers round-robin.

    Episode ``k`` goes to worker ``k % n`` so every shard contains the same
    initial states as the serial path (init states are indexed by absolute
    episode number). Shards are therefore result-identical to a serial run.
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be >= 1")
    if num_episodes <= 0:
        raise ValueError(
            f"num_episodes must be >= 1, got {num_episodes} — refusing to "
            "produce zero-size shards"
        )
    n = min(num_workers, num_episodes)
    return [list(range(w, num_episodes, n)) for w in range(n)]


def _merge_worker_results(
    worker_results: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Aggregate per-task success counts from parallel workers.

    Returns ``{task_id: {"description": str, "successes": int}}`` — the
    same shape the serial path builds — so both feed
    :func:`_finalize_suite_results` identically. Task IDs are primary keys
    (E9); descriptions are merged from the first worker that saw the task.
    """
    merged: dict[int, dict[str, Any]] = {}
    for payload in worker_results:
        for task_id, counts in payload["task_results"].items():
            key = int(task_id)
            entry = merged.setdefault(key, {"description": "", "successes": 0})
            entry["successes"] += int(counts["successes"])
            if counts.get("description"):
                entry["description"] = counts["description"]
    return merged


def _finalize_suite_results(
    suite_name: str,
    per_task_successes: dict[int, dict[str, Any]],
    num_episodes_per_task: int,
    expected_tasks: int | None = None,
) -> dict[str, Any]:
    """Convert per-task success counts into the standard suite result dict.

    Shared by the serial and parallel paths so both produce identical
    ``eval/*`` result structures. Per-task entries are keyed by numeric
    task ID with the description retained as metadata (E9).

    Args:
        expected_tasks: When given, the number of per-task entries is
            asserted against it (runtime task-count check, E9).
    """
    if expected_tasks is not None and len(per_task_successes) != expected_tasks:
        raise ValueError(
            f"{suite_name}: expected {expected_tasks} per-task entries, got "
            f"{len(per_task_successes)} — task ids are misaligned or a task "
            "was evaluated twice/never."
        )
    if num_episodes_per_task < 1:
        raise ValueError(
            f"{suite_name}: num_episodes_per_task must be >= 1, got "
            f"{num_episodes_per_task}"
        )
    per_task_results: dict[str, dict[str, Any]] = {}
    total_episodes = 0
    total_successes = 0
    for task_id, entry in sorted(per_task_successes.items()):
        successes = int(entry["successes"])
        if successes < 0 or successes > num_episodes_per_task:
            raise ValueError(
                f"{suite_name} task {task_id}: successes={successes} is outside "
                f"[0, {num_episodes_per_task}]"
            )
        rate = (
            successes / num_episodes_per_task if num_episodes_per_task else 0.0
        )
        per_task_results[str(task_id)] = {
            "description": entry.get("description", ""),
            "success_rate": rate,
            "successes": successes,
            "episodes": num_episodes_per_task,
        }
        total_episodes += num_episodes_per_task
        total_successes += successes

    overall_success_rate = (
        total_successes / total_episodes if total_episodes > 0 else 0.0
    )
    return {
        f"eval/success_rate/{suite_name}": overall_success_rate,
        f"eval/per_task/{suite_name}": per_task_results,
        f"eval/total_episodes/{suite_name}": total_episodes,
        f"eval/total_successes/{suite_name}": total_successes,
    }


def _run_suite_worker(
    cfg: DictConfig,
    suite_name: str,
    num_tasks: int,
    episode_indices: list[int],
    worker_idx: int,
    num_workers: int,
    result_queue: multiprocessing.Queue,
) -> None:
    """Evaluate one episode shard of a suite inside a spawned process.

    Each worker rebuilds the model from the same config/checkpoint (via
    :func:`phaseforge.cli.build_eval_model`) so nothing GPU-bound is
    pickled across processes. GPU memory usage scales roughly linearly
    with worker count — keep ``num_workers`` modest on small VRAM cards.

    Results are sent as ``{"worker_idx": int, "task_results": {task_id: {…}}}``.
    A crash inside the worker propagates as a nonzero process exit; the
    parent raises a RuntimeError after :meth:`join`.
    """
    # Spawned processes start with an unconfigured root logger, so INFO
    # progress lines (task/episode logs below) would be silently dropped.
    # Mirror the main-process format so worker progress appears on the
    # same console.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    from phaseforge.cli import _resolve_device, build_eval_model
    from phaseforge.utils.seed import set_seed

    set_seed(cfg.project.seed + worker_idx)
    torch.set_num_threads(1)  # avoid OpenMP oversubscription vs MuJoCo
    device = _resolve_device(cfg)
    model = build_eval_model(cfg)
    model.to(device)
    evaluator = RolloutEvaluator(cfg=cfg, model=model, device=device)

    logger.info(
        "Worker %d/%d: evaluating suite %s — %d task(s), %d episode(s): %s",
        worker_idx + 1, num_workers, suite_name, num_tasks,
        len(episode_indices), episode_indices,
    )
    task_results: dict[int, dict[str, Any]] = {}
    for task_id in range(num_tasks):
        task_id_out, task_desc, successes = evaluator._evaluate_task(
            suite_name, task_id, episode_indices, num_tasks
        )
        task_results[task_id_out] = {
            "description": task_desc,
            "successes": successes,
            "episodes": len(episode_indices),
        }
    result_queue.put(
        {
            "worker_idx": worker_idx,
            "task_results": task_results,
            # The parent cannot inspect an environment created in this
            # spawned worker. Return controller provenance explicitly.
            "controller_meta": evaluator._controller_meta,
        }
    )


def _sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a checkpoint file (metadata, not full read)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _library_versions() -> dict[str, str]:
    """Best-effort versions of the environment stack (E9 metadata)."""
    versions: dict[str, str] = {}
    for lib in ("libero", "robosuite", "mujoco", "gymnasium"):
        try:
            module = __import__(lib)
            versions[lib] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001 — metadata is best-effort
            versions[lib] = "not installed"
    return versions


class RolloutEvaluator:
    """Environment-based rollout evaluator for LIBERO.

    Normalizes environment states with the training-frozen normalizer,
    runs model inference via ``get_action()``, and aggregates per-suite
    and per-task success rates.

    Settings are read defensively so that setting ``eval.mode=rollout``
    without the full rollout.yaml still works:
        - ``eval.environment.suites`` (default ``["libero_90"]`` —
          the in-distribution core per Decision 2, issues register A2;
          spatial/object/goal are NOT valid fallbacks)
        - ``eval.environment.num_steps_wait`` (default 10)
        - ``eval.environment.render_observations`` (default False)
        - ``eval.environment.hard_reset`` (default True: rebuild the sim
          from XML every episode — the official protocol, bit-identical)
        - ``eval.environment.num_workers`` (default 0 = auto: one per CPU)
        - ``eval.environment.object_state`` (default None = disabled)
        - ``eval.evaluation.num_episodes_per_task`` (default 50)
        - ``eval.evaluation.episodes_per_suite`` (default {} = use the
          per-task count for every suite; E5 sets libero_90=50, libero_10=10)
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.model.eval()
        self.normalizer = self._load_normalizer()
        self._controller_meta: dict[str, Any] | None = None
        self._action_policy_recorded = False

        # Resolve rollout settings defensively so that setting
        # ``eval.mode=rollout`` without the full rollout.yaml still works.
        eval_cfg = cfg.eval
        env_cfg = eval_cfg.get("environment", None) or {}
        eval_settings = eval_cfg.get("evaluation", None) or {}
        self.suites: list[str] = list(env_cfg.get("suites", ["libero_90"]))
        self.num_steps_wait: int = int(env_cfg.get("num_steps_wait", 10))
        self.render_observations: bool = bool(
            env_cfg.get("render_observations", False)
        )
        self.hard_reset: bool = bool(env_cfg.get("hard_reset", True))
        self.num_workers: int = int(env_cfg.get("num_workers", 0))
        self.num_episodes_per_task: int = int(
            eval_settings.get("num_episodes_per_task", 50)
        )
        # Per-suite episode counts (E5): libero_90 = 50 eps/task (ID),
        # libero_10 = 10 eps/task (labeled zero-shot row). Falls back to
        # ``num_episodes_per_task`` for suites without an entry.
        self.episodes_per_suite: dict[str, int] = {
            str(k): int(v) for k, v in (eval_settings.get("episodes_per_suite") or {}).items()
        }
        # P-Stage 1 object-state channel (None = disabled).
        self.object_state_cfg: Any = env_cfg.get("object_state")

        self._validate_settings()

    def _validate_settings(self) -> None:
        """Fail fast on protocol-breaking settings (issues register E9).

        A zero episode count would otherwise surface as an obscure
        ``range(... step=0)`` error deep inside shard splitting.
        """
        # Only suites with a complete protocol definition (benchmark name,
        # expected task count, max steps) are runnable. ``libero_long`` is a
        # legacy alias of the libero_10 benchmark that has NO protocol entry
        # (no SUITE_N_TASKS/SUITE_MAX_STEPS mapping) — rejecting it here
        # beats a KeyError halfway through evaluation.
        supported = sorted(
            set(SUITE_BENCHMARK_NAMES) & set(SUITE_N_TASKS) & set(SUITE_MAX_STEPS)
        )
        unknown = [s for s in self.suites if s not in supported]
        if unknown:
            raise ValueError(
                f"Unknown or unsupported suite name(s): {unknown}. Supported "
                f"suites: {supported}. ('libero_long' is a legacy alias with "
                "no protocol definition — use 'libero_10'.)"
            )
        if self.num_episodes_per_task < 1:
            raise ValueError(
                f"num_episodes_per_task must be >= 1, got "
                f"{self.num_episodes_per_task}"
            )
        for suite, count in self.episodes_per_suite.items():
            if suite not in self.suites:
                raise ValueError(
                    f"episodes_per_suite contains {suite!r}, but that suite is "
                    f"not in eval.environment.suites={self.suites}"
                )
            if count < 1:
                raise ValueError(
                    f"episodes_per_suite[{suite}] must be >= 1, got {count}"
                )
        if self.num_steps_wait < 0:
            raise ValueError(
                f"num_steps_wait must be >= 0, got {self.num_steps_wait}"
            )
        if self.num_workers < 0:
            raise ValueError(
                f"num_workers must be >= 0 (0 = auto), got {self.num_workers}"
            )

    def _episodes_for_suite(self, suite_name: str) -> int:
        return self.episodes_per_suite.get(suite_name, self.num_episodes_per_task)

    def _load_normalizer(self) -> FrozenNormalizer:
        """Load the training-frozen normalizer from the processed cache.

        Uses the same cache-hash mechanism as ``DataPipelineStateMachine``
        so the normalizer is guaranteed to match training-time statistics.
        The cache hash and manifest provenance are recorded for the eval
        payload (issues register E9, items 14/16).
        """
        cache_root = processed_cache_root()
        cache_mgr = CacheManager(cache_root)
        self._cache_hash = CacheManager.compute_hash(self.cfg.data)
        
        enforce_strict = self.cfg.data.get("enforce_strict_cache", True)
        found_hash = cache_mgr.find_cache(self._cache_hash, enforce_strict)
        
        if found_hash:
            self._cache_hash = found_hash
            try:
                _, norm_stats, _, _ = cache_mgr.load(self._cache_hash)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Cache corrupted for hash {self._cache_hash}"
                ) from exc
        else:
            raise RuntimeError(
                f"No cached dataset found for config_hash={self._cache_hash}. "
                "Run training (which builds the cache) before rollout evaluation."
            )
        # Manifest provenance: object-index hash, git commit, state schema —
        # recorded so the eval result can be audited against the cache.
        manifest_path = cache_mgr.cache_dir(self._cache_hash) / "manifest.json"
        self._cache_manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                self._cache_manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    "Could not read cache manifest %s — provenance will be "
                    "incomplete in the eval payload.",
                    manifest_path,
                )
        normalizer = FrozenNormalizer(mean=norm_stats["mean"], std=norm_stats["std"])
        # Pin the stats to the eval device once. ``normalize()`` re-hosts
        # mean/std on every call (normalizer.py:83-84); doing it here keeps
        # that per-step work out of the rollout hot loop.
        normalizer.mean = normalizer.mean.to(self.device)
        normalizer.std = normalizer.std.to(self.device)
        return normalizer

    @torch.inference_mode()
    def _get_action(self, state: np.ndarray) -> np.ndarray:
        """Run model inference: state (S,) -> action (7,).

        Normalizes the state, calls ``get_action()``, and validates the
        action (exact 7 dims, finite values) before returning it. Actions
        are NEVER silently clipped; a violation raises immediately so an
        invalid policy cannot silently poison a whole suite (E9).

        ``inference_mode`` is safe here: ``get_action()`` is a pure
        forward pass (no autograd hooks or in-place leaf mutation), and it
        is measurably cheaper than ``no_grad`` on long rollout loops.
        """
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        state_tensor = self.normalizer.normalize(state_tensor)
        action = self.model.get_action(state_tensor)
        if action.ndim == 2:
            action = action.squeeze(0)
        self._validate_action(action)
        return action.cpu().numpy().astype(np.float64)

    def _validate_action(self, action: torch.Tensor) -> None:
        """Validate the model output before it reaches the environment.

        Raises:
            ValueError: If the action is not exactly ``action_dim``-shaped,
                or contains non-finite values. Clipping is deliberately
                never applied.
        """
        action_dim = int(self.cfg.data.get("action_dim", ACTION_DIM_DEFAULT))
        if action.ndim != 1 or action.shape[0] != action_dim:
            raise ValueError(
                f"Model returned action of shape {tuple(action.shape)} — "
                f"expected ({action_dim},). The policy's action space does "
                "not match the LIBERO OSC_POSE convention; refusing to "
                "run a mismatched controller."
            )
        if not torch.isfinite(action).all():
            raise ValueError(
                f"Model returned non-finite action values: "
                f"{action.detach().cpu().numpy()}. Refusing to step the env "
                "with NaN/Inf actions (they would silently corrupt the "
                "physics state)."
            )
        if not self._action_policy_recorded:
            self._action_policy_recorded = True
            logger.info(
                "Action validation active: shape=(%d,), finite=required, "
                "clip=none (no silent clipping).",
                action_dim,
            )

    def _validate_state(self, state: np.ndarray) -> None:
        """Validate an env state vector against the normalizer/model input.

        Raises:
            ValueError: On dimensionality mismatch (train/eval schema
                drift, issues register B7/E9) or non-finite values.
        """
        expected = int(self.normalizer.mean.shape[0])
        if state.ndim != 1 or state.shape[0] != expected:
            raise ValueError(
                f"Env returned state of shape {getattr(state, 'shape', None)} "
                f"— expected ({expected},) from the training normalizer. "
                "Train/eval state schema has drifted; fix the state keys or "
                "object index before trusting any rollout."
            )
        if not np.isfinite(state).all():
            raise ValueError("Env returned non-finite state values.")

    def _capture_controller_metadata(self, env: StateOnlyLiberoEnv) -> None:
        """Record the installed controller identity (action semantics, E9).

        The training HDF5 actions were generated by LIBERO's default
        OffScreenRenderEnv controller (OSC_POSE, delta EEF pose + gripper).
        This records the ACTUAL controller of the installed env so the
        train/eval action-convention equivalence is auditable, and logs a
        warning when the installed controller is not OSC_POSE.
        """
        if self._controller_meta is not None:
            return
        controller: Any = None
        try:
            env_attr = getattr(env, "_env", None)
            robots = getattr(env_attr, "robots", None)
            if robots:
                controller = robots[0].controller
        except Exception:  # noqa: BLE001 — metadata is best-effort
            controller = None
        if controller is None:
            self._controller_meta = {"note": "controller introspection unavailable"}
            return
        self._controller_meta = {
            "name": getattr(controller, "name", "unknown"),
            "control_freq": getattr(controller, "control_freq", None),
            "action_scale": getattr(controller, "action_scale", None),
            "interpolation": getattr(controller, "interpolation", None),
        }
        if self._controller_meta.get("name") != "OSC_POSE":
            logger.warning(
                "Installed env controller is %r (expected OSC_POSE) — the "
                "7-DoF action semantics of the training HDF5 data are only "
                "guaranteed for the default LIBERO controller. Recorded in "
                "eval/controller; inspect before reporting numbers.",
                self._controller_meta.get("name"),
            )

    def _evaluate_task(
        self,
        suite_name: str,
        task_id: int,
        episode_indices: list[int],
        num_tasks: int,
    ) -> tuple[int, str, int]:
        """Run the given absolute episode indices on one task.

        Used by both the serial path (all episodes) and the parallel
        workers (a round-robin shard). Returns ``(task_id, task_description,
        successes)`` — the task ID is the primary key (E9).
        """
        env = StateOnlyLiberoEnv(
            suite_name=suite_name,
            task_id=task_id,
            seed=self.cfg.project.seed,
            num_steps_wait=self.num_steps_wait,
            render_observations=self.render_observations,
            hard_reset=self.hard_reset,
            object_state_cfg=self.object_state_cfg,
        )
        task_desc = env.task_description
        max_steps = SUITE_MAX_STEPS[suite_name]
        task_successes = 0
        self._capture_controller_metadata(env)

        # Initial-state availability check (E9): every requested episode
        # index must be a valid init state, or we abort the task instead of
        # raising a confusing per-episode IndexError halfway through.
        num_init_states = env.num_init_states
        if episode_indices and max(episode_indices) >= num_init_states:
            env.close()
            raise IndexError(
                f"Task {task_id} ({task_desc}): requested episode index "
                f"{max(episode_indices)} but only {num_init_states} initial "
                f"states are available — reduce num_episodes_per_task / "
                "episodes_per_suite."
            )

        logger.info(
            "  [%d/%d] task: %s — running %d episodes",
            task_id + 1, num_tasks, task_desc, len(episode_indices),
        )

        try:
            for shard_pos, episode_idx in enumerate(episode_indices):
                state = env.reset(episode_idx=episode_idx)
                self._validate_state(state)
                episode_success = False

                for _ in range(max_steps):
                    action = self._get_action(state)
                    state, _, terminated, truncated, info = env.step(action)
                    self._validate_state(state)

                    if terminated or truncated:
                        if info.get("is_success", False):
                            episode_success = True
                        break

                if episode_success:
                    task_successes += 1

                # Log progress every 10 episodes of this shard
                if (shard_pos + 1) % 10 == 0:
                    logger.info(
                        "    episode %d/%d — running SR: %.1f%%",
                        shard_pos + 1, len(episode_indices),
                        100.0 * task_successes / (shard_pos + 1),
                    )
        finally:
            env.close()

        return task_id, task_desc, task_successes

    def evaluate_suite(
        self,
        suite_name: str,
        num_episodes_per_task: int,
    ) -> dict[str, Any]:
        """Evaluate a single LIBERO suite.

        Dispatches to the serial path (``num_workers <= 1``) or the
        parallel worker-sharded path. Returns per-task success rates and
        suite-level aggregates in the same format either way.
        """
        bench_key = SUITE_BENCHMARK_NAMES[suite_name]

        # Prevent LIBERO interactive prompt hanging in non-interactive environments
        from pathlib import Path
        libero_cfg = Path.home() / ".libero" / "config.yaml"
        if not libero_cfg.exists():
            libero_cfg.parent.mkdir(parents=True, exist_ok=True)
            libero_cfg.write_text("DATASET_PATH: ''\n")

        try:
            from libero.libero import benchmark
        except ImportError as exc:
            raise RuntimeError(
                "libero package not installed. Run: pip install libero"
            ) from exc

        task_suite = benchmark.get_benchmark_dict()[bench_key]()
        num_tasks = task_suite.n_tasks
        # Runtime task-count assertion (E9): the summary code assumes fixed
        # suite sizes; a version mismatch must abort loudly, not silently
        # evaluate a different task grid.
        expected_tasks = SUITE_N_TASKS[suite_name]
        if num_tasks != expected_tasks:
            raise RuntimeError(
                f"Suite {suite_name}: installed LIBERO reports {num_tasks} "
                f"tasks but the protocol expects {expected_tasks}. The "
                "installed LIBERO/robosuite version does not match the "
                "protocol definition — do not evaluate against it."
            )
        num_workers = _resolve_num_workers(self.num_workers, num_episodes_per_task)

        suite_start = time.monotonic()

        if num_workers <= 1:
            per_task_successes: dict[int, dict[str, Any]] = {}
            for task_id in range(num_tasks):
                task_start = time.monotonic()
                task_id_out, task_desc, task_successes = self._evaluate_task(
                    suite_name,
                    task_id,
                    list(range(num_episodes_per_task)),
                    num_tasks,
                )
                per_task_successes[task_id_out] = {
                    "description": task_desc,
                    "successes": task_successes,
                }
                elapsed = time.monotonic() - task_start
                logger.info(
                    "  [%d/%d] task: %s — SR: %.1f%% (%d/%d) [%.0fs]",
                    task_id + 1, num_tasks, task_desc,
                    100.0 * task_successes / max(num_episodes_per_task, 1),
                    task_successes, num_episodes_per_task, elapsed,
                )
        else:
            per_task_successes = self._evaluate_suite_parallel(
                suite_name, num_tasks, num_episodes_per_task, num_workers
            )

        results = _finalize_suite_results(
            suite_name, per_task_successes, num_episodes_per_task,
            expected_tasks=expected_tasks,
        )
        suite_elapsed = time.monotonic() - suite_start
        logger.info(
            "Suite %s finished: SR=%.1f%% (%d/%d) [%.0fs]",
            suite_name,
            100.0 * results[f"eval/success_rate/{suite_name}"],
            results[f"eval/total_successes/{suite_name}"],
            results[f"eval/total_episodes/{suite_name}"],
            suite_elapsed,
        )
        return results

    def _evaluate_suite_parallel(
        self,
        suite_name: str,
        num_tasks: int,
        num_episodes_per_task: int,
        num_workers: int,
    ) -> dict[int, dict[str, Any]]:
        """Shard episodes across spawned worker processes and aggregate.

        Workers are spawned with the ``spawn`` context (required for
        MuJoCo/EGL + LIBERO on Linux; also the safe default on Windows).
        A worker crash surfaces as a nonzero exit code and aborts the
        suite rather than silently producing partial results.
        """
        shards = split_episode_shards(num_episodes_per_task, num_workers)
        ctx = multiprocessing.get_context("spawn")
        result_queue: multiprocessing.Queue = ctx.Queue()
        processes = []
        for worker_idx, episode_indices in enumerate(shards):
            proc = ctx.Process(
                target=_run_suite_worker,
                args=(
                    self.cfg,
                    suite_name,
                    num_tasks,
                    episode_indices,
                    worker_idx,
                    num_workers,
                    result_queue,
                ),
            )
            proc.start()
            processes.append(proc)

        logger.info(
            "Suite %s: %d worker(s) spawned — episode shards: %s",
            suite_name, num_workers, [len(s) for s in shards],
        )

        for proc in processes:
            proc.join()

        failed = [p for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(
                f"{len(failed)} evaluation worker(s) failed "
                f"(exit codes: {[p.exitcode for p in failed]}). "
                "See worker stderr for details."
            )

        worker_results = [result_queue.get() for _ in processes]
        controller_meta = next(
            (
                result.get("controller_meta")
                for result in worker_results
                if result.get("controller_meta") is not None
            ),
            None,
        )
        if controller_meta is not None:
            self._controller_meta = controller_meta
        return _merge_worker_results(worker_results)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        """Checkpoint path + SHA-256 + its training run_meta (E9, items 14/16).

        Also warns when the training run's git commit differs from the
        current commit (train/eval code drift) — recorded, never fatal.
        """
        meta: dict[str, Any] = {"path": None, "sha256": None, "run_meta": None}
        train_cfg = self.cfg.get("train")
        ckpt_path = train_cfg.get("stage1_ckpt_path") if train_cfg is not None else None
        if not ckpt_path:
            return meta
        ckpt = Path(ckpt_path).resolve()
        meta["path"] = str(ckpt)
        try:
            meta["sha256"] = _sha256_file(ckpt)
        except OSError as exc:
            logger.warning("Could not hash checkpoint %s: %s", ckpt, exc)
        run_meta_path = ckpt.parent.parent / "run_meta.json"
        if run_meta_path.is_file():
            try:
                meta["run_meta"] = json.loads(run_meta_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Could not read run_meta.json next to checkpoint %s: %s",
                    ckpt, exc,
                )
            else:
                train_commit = meta["run_meta"].get("git_commit") or ""
                cur_commit = self._git_commit
                if train_commit and cur_commit and train_commit != cur_commit:
                    logger.warning(
                        "Checkpoint %s was trained at git commit %s but the "
                        "current commit is %s — train/eval code may differ. "
                        "Recorded in eval/checkpoint.run_meta.",
                        ckpt, train_commit, cur_commit,
                    )
        return meta

    def _provenance(self) -> dict[str, Any]:
        """Full provenance payload (issues register E9, items 14/16/17)."""
        cfg = self.cfg
        git = git_info()
        self._git_commit = git.get("commit", "")
        manifest_prov = self._cache_manifest.get("provenance") or {}

        state_repr: dict[str, Any] = {
            "state_dim": int(self.normalizer.mean.shape[0]),
            "proprio_dim": 23,
        }
        if self.object_state_cfg is not None:
            oscfg = (
                self.object_state_cfg
                if isinstance(self.object_state_cfg, DictConfig)
                else self.object_state_cfg
            )
            state_repr.update(
                {
                    "object_state": {
                        "enabled": bool(oscfg.get("enabled", True)),
                        "k_slots": int(oscfg.get("k_slots", 0)),
                        "dim_per_object": int(oscfg.get("dim_per_object", 0)),
                        "include_mask": bool(oscfg.get("include_mask", False)),
                    }
                }
            )

        action_policy = {
            "dim": int(cfg.data.get("action_dim", ACTION_DIM_DEFAULT)),
            "validation": (
                "raise on wrong shape or NaN/Inf — actions are never "
                "silently clipped"
            ),
            "clip": "none",
            "convention": (
                "OSC_POSE delta EEF pose + gripper (LIBERO OffScreenRenderEnv "
                "default controller — the same 7-DoF convention the training "
                "HDF5 actions were generated with); installed controller "
                "recorded under eval/controller"
            ),
        }

        episodes_per_suite = {
            suite: self._episodes_for_suite(suite) for suite in self.suites
        }

        data_hash = CacheManager.compute_hash(cfg.data)
        return {
            "eval/checkpoint": self._checkpoint_metadata(),
            "eval/git_commit": git.get("commit", ""),
            "eval/git_branch": git.get("branch", ""),
            "eval/config_hash": CacheManager.compute_hash(cfg),
            "eval/data_config_hash": data_hash,
            "eval/cache_hash": self._cache_hash,
            "eval/cache_object_index_sha256": (
                manifest_prov.get("object_index_sha256")
            ),
            "eval/cache_git_commit": manifest_prov.get("git_commit"),
            "eval/state_repr": state_repr,
            "eval/action_policy": action_policy,
            "eval/controller": self._controller_meta,
            "eval/num_workers_requested": self.num_workers,
            "eval/num_workers_effective": dict(self._effective_workers_by_suite),
            "eval/max_steps_per_suite": {
                suite: SUITE_MAX_STEPS[suite] for suite in self.suites
            },
            "eval/num_steps_wait": self.num_steps_wait,
            "eval/hard_reset": self.hard_reset,
            "eval/render_observations": self.render_observations,
            "eval/episodes_per_suite": episodes_per_suite,
            "eval/versions": _library_versions(),
        }

    def run(self) -> dict[str, Any]:
        """Run rollout evaluation across all configured suites.

        Returns:
            Dict with per-suite success rates, per-task breakdowns,
            an overall average across suites, and full provenance metadata.
        """
        all_results: dict[str, Any] = {}
        all_suite_rates: list[float] = []
        self._effective_workers_by_suite: dict[str, int] = {}

        resolved_workers = _resolve_num_workers(
            self.num_workers, max(self.num_episodes_per_task, 1)
        )
        if self.num_workers <= 0:
            logger.info(
                "num_workers=auto: resolved to %d worker(s) "
                "(%d logical CPU(s))",
                resolved_workers, os.cpu_count() or 1,
            )

        for suite_name in self.suites:
            episodes_per_task = self._episodes_for_suite(suite_name)
            effective_workers = _resolve_num_workers(
                self.num_workers, episodes_per_task
            )
            self._effective_workers_by_suite[suite_name] = effective_workers
            logger.info(
                "Evaluating suite %s (%d episodes/task, %d worker(s))…",
                suite_name,
                episodes_per_task,
                effective_workers,
            )
            suite_results = self.evaluate_suite(
                suite_name, num_episodes_per_task=episodes_per_task
            )
            all_results.update(suite_results)
            rate_key = f"eval/success_rate/{suite_name}"
            if rate_key in suite_results:
                all_suite_rates.append(float(suite_results[rate_key]))

        if all_suite_rates:
            suite_weights = {
                suite: self._episodes_for_suite(suite) for suite in self.suites
            }
            total_weight = sum(suite_weights.values())
            all_results["eval/success_rate"] = float(
                sum(
                    float(all_results[f"eval/success_rate/{suite}"])
                    * suite_weights[suite]
                    for suite in self.suites
                    if f"eval/success_rate/{suite}" in all_results
                )
                / total_weight
            )
            all_results["eval/overall_aggregation"] = (
                "episode-weighted mean of suite success rates"
            )

        all_results["eval/seed"] = self.cfg.project.seed
        all_results["eval/num_episodes_per_task"] = self.num_episodes_per_task
        all_results["eval/suites"] = list(self.suites)
        # B4: ID-vs-OOD declaration — no suite is implicitly ID.
        all_results["eval/suite_roles"] = {
            suite: SUITE_ID_OOD_ROLES.get(suite, "unclassified")
            for suite in self.suites
        }
        # E9: per-suite episode counts + full provenance (checkpoint hash,
        # git commit, config/data/cache hashes, versions, controller,
        # action policy, effective settings).
        all_results.update(self._provenance())

        return all_results
