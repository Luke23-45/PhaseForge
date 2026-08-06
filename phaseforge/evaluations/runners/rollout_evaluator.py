"""Rollout evaluator using LIBERO environment.

Runs the standard evaluation protocol:
- For each task in each configured suite
- For each episode, reset env to initial state, run policy until done/max_steps
- Record binary success from env.check_success()
- Aggregate per-suite and overall success rates

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

import logging
import multiprocessing
import os
import time
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

logger = logging.getLogger(__name__)


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
    n = min(num_workers, num_episodes)
    return [list(range(w, num_episodes, n)) for w in range(n)]


def _merge_worker_results(
    worker_results: list[dict[str, Any]],
) -> dict[str, int]:
    """Aggregate per-task success counts from parallel workers.

    Returns ``{task_description: total_successes}`` — the same shape the
    serial path builds — so both feed :func:`_finalize_suite_results`
    identically.
    """
    merged: dict[str, int] = {}
    for payload in worker_results:
        for task_desc, counts in payload["task_results"].items():
            merged[task_desc] = merged.get(task_desc, 0) + int(counts["successes"])
    return merged


def _finalize_suite_results(
    suite_name: str,
    per_task_successes: dict[str, int],
    num_episodes_per_task: int,
) -> dict[str, Any]:
    """Convert per-task success counts into the standard suite result dict.

    Shared by the serial and parallel paths so both produce identical
    ``eval/*`` result structures.
    """
    per_task_results: dict[str, dict[str, float | int]] = {}
    total_episodes = 0
    total_successes = 0
    for task_desc, successes in per_task_successes.items():
        rate = (
            successes / num_episodes_per_task if num_episodes_per_task else 0.0
        )
        per_task_results[task_desc] = {
            "success_rate": rate,
            "successes": int(successes),
            "episodes": num_episodes_per_task,
        }
        total_episodes += num_episodes_per_task
        total_successes += int(successes)

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

    Results are sent as ``{"worker_idx": int, "task_results": {desc: {…}}}``.
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
    task_results: dict[str, dict[str, int]] = {}
    for task_id in range(num_tasks):
        task_desc, successes = evaluator._evaluate_task(
            suite_name, task_id, episode_indices, num_tasks
        )
        task_results[task_desc] = {
            "successes": successes,
            "episodes": len(episode_indices),
        }
    result_queue.put({"worker_idx": worker_idx, "task_results": task_results})


class RolloutEvaluator:
    """Environment-based rollout evaluator for LIBERO.

    Normalizes environment states with the training-frozen normalizer,
    runs model inference via ``get_action()``, and aggregates per-suite
    and per-task success rates.

    Settings are read defensively so that setting ``eval.mode=rollout``
    without the full rollout.yaml still works:
        - ``eval.environment.suites`` (default ``["libero_spatial"]``)
        - ``eval.environment.num_steps_wait`` (default 10)
        - ``eval.environment.render_observations`` (default False)
        - ``eval.environment.num_workers`` (default 0 = auto: one per CPU)
        - ``eval.evaluation.num_episodes_per_task`` (default 50)
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

        # Resolve rollout settings defensively so that setting
        # ``eval.mode=rollout`` without the full rollout.yaml still works.
        eval_cfg = cfg.eval
        env_cfg = eval_cfg.get("environment", None) or {}
        eval_settings = eval_cfg.get("evaluation", None) or {}
        self.suites: list[str] = list(env_cfg.get("suites", ["libero_spatial"]))
        self.num_steps_wait: int = int(env_cfg.get("num_steps_wait", 10))
        self.render_observations: bool = bool(
            env_cfg.get("render_observations", False)
        )
        self.num_workers: int = int(env_cfg.get("num_workers", 0))
        self.num_episodes_per_task: int = int(
            eval_settings.get("num_episodes_per_task", 50)
        )

    def _load_normalizer(self) -> FrozenNormalizer:
        """Load the training-frozen normalizer from the processed cache.

        Uses the same cache-hash mechanism as ``DataPipelineStateMachine``
        so the normalizer is guaranteed to match training-time statistics.
        """
        cache_root = processed_cache_root()
        cache_mgr = CacheManager(cache_root)
        config_hash = CacheManager.compute_hash(self.cfg.data)
        try:
            _, norm_stats, _, _ = cache_mgr.load(config_hash)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"No cached dataset found for config_hash={config_hash}. "
                "Run training (which builds the cache) before rollout evaluation."
            ) from exc
        return FrozenNormalizer(mean=norm_stats["mean"], std=norm_stats["std"])

    @torch.inference_mode()
    def _get_action(self, state: np.ndarray) -> np.ndarray:
        """Run model inference: state (23,) -> action (7,).

        Normalizes the state, calls ``get_action()``, and returns
        the raw (already unnormalized) action as a numpy array.

        ``inference_mode`` is safe here: ``get_action()`` is a pure
        forward pass (no autograd hooks or in-place leaf mutation), and it
        is measurably cheaper than ``no_grad`` on long rollout loops.
        """
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        state_tensor = self.normalizer.normalize(state_tensor)
        action = self.model.get_action(state_tensor)
        if action.ndim == 2:
            action = action.squeeze(0)
        return action.cpu().numpy().astype(np.float64)

    def _evaluate_task(
        self,
        suite_name: str,
        task_id: int,
        episode_indices: list[int],
        num_tasks: int,
    ) -> tuple[str, int]:
        """Run the given absolute episode indices on one task.

        Used by both the serial path (all episodes) and the parallel
        workers (a round-robin shard). Returns ``(task_description,
        successes)``.
        """
        env = StateOnlyLiberoEnv(
            suite_name=suite_name,
            task_id=task_id,
            seed=self.cfg.project.seed,
            num_steps_wait=self.num_steps_wait,
            render_observations=self.render_observations,
        )
        task_desc = env.task_description
        max_steps = SUITE_MAX_STEPS[suite_name]
        task_successes = 0

        logger.info(
            "  [%d/%d] task: %s — running %d episodes",
            task_id + 1, num_tasks, task_desc, len(episode_indices),
        )

        try:
            for shard_pos, episode_idx in enumerate(episode_indices):
                state = env.reset(episode_idx=episode_idx)
                episode_success = False

                for _ in range(max_steps):
                    action = self._get_action(state)
                    state, _, terminated, truncated, info = env.step(action)

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

        return task_desc, task_successes

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

        try:
            from libero.libero import benchmark
        except ImportError as exc:
            raise RuntimeError(
                "libero package not installed. Run: pip install libero"
            ) from exc

        task_suite = benchmark.get_benchmark_dict()[bench_key]()
        num_tasks = task_suite.n_tasks
        num_workers = _resolve_num_workers(self.num_workers, num_episodes_per_task)

        suite_start = time.monotonic()

        if num_workers <= 1:
            per_task_successes: dict[str, int] = {}
            for task_id in range(num_tasks):
                task_start = time.monotonic()
                task_desc, task_successes = self._evaluate_task(
                    suite_name,
                    task_id,
                    list(range(num_episodes_per_task)),
                    num_tasks,
                )
                per_task_successes[task_desc] = task_successes
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
            suite_name, per_task_successes, num_episodes_per_task
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
    ) -> dict[str, int]:
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
        return _merge_worker_results(worker_results)

    def run(self) -> dict[str, Any]:
        """Run rollout evaluation across all configured suites.

        Returns:
            Dict with per-suite success rates, per-task breakdowns,
            and an overall average across suites.
        """
        all_results: dict[str, Any] = {}
        all_suite_rates: list[float] = []

        resolved_workers = _resolve_num_workers(
            self.num_workers, self.num_episodes_per_task
        )
        if self.num_workers <= 0:
            logger.info(
                "num_workers=auto: resolved to %d worker(s) "
                "(%d logical CPU(s))",
                resolved_workers, os.cpu_count() or 1,
            )

        for suite_name in self.suites:
            logger.info(
                "Evaluating suite %s (%d episodes/task, %d worker(s))…",
                suite_name,
                self.num_episodes_per_task,
                resolved_workers,
            )
            suite_results = self.evaluate_suite(
                suite_name, num_episodes_per_task=self.num_episodes_per_task
            )
            all_results.update(suite_results)
            rate_key = f"eval/success_rate/{suite_name}"
            if rate_key in suite_results:
                all_suite_rates.append(float(suite_results[rate_key]))

        if all_suite_rates:
            all_results["eval/success_rate"] = float(np.mean(all_suite_rates))

        all_results["eval/seed"] = self.cfg.project.seed
        all_results["eval/num_episodes_per_task"] = self.num_episodes_per_task
        all_results["eval/suites"] = list(self.suites)

        return all_results
