"""Rollout evaluator using LIBERO environment.

Runs the standard evaluation protocol:
- For each task in each configured suite
- For each episode, reset env to initial state, run policy until done/max_steps
- Record binary success from env.check_success()
- Aggregate per-suite and overall success rates
"""

from __future__ import annotations

import logging
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


class RolloutEvaluator:
    """Environment-based rollout evaluator for LIBERO.

    Normalizes environment states with the training-frozen normalizer,
    runs model inference via ``get_action()``, and aggregates per-suite
    and per-task success rates.
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

    @torch.no_grad()
    def _get_action(self, state: np.ndarray) -> np.ndarray:
        """Run model inference: state (23,) -> action (7,).

        Normalizes the state, calls ``get_action()``, and returns
        the raw (already unnormalized) action as a numpy array.
        """
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        state_tensor = self.normalizer.normalize(state_tensor)
        action = self.model.get_action(state_tensor)
        if action.ndim == 2:
            action = action.squeeze(0)
        return action.cpu().numpy().astype(np.float64)

    def evaluate_suite(
        self,
        suite_name: str,
        num_episodes_per_task: int,
    ) -> dict[str, Any]:
        """Evaluate a single LIBERO suite.

        Returns per-task success rates and suite-level aggregates.
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

        total_episodes = 0
        total_successes = 0
        per_task_results: dict[str, dict[str, float | int]] = {}

        for task_id in range(num_tasks):
            env = StateOnlyLiberoEnv(
                suite_name=suite_name,
                task_id=task_id,
                seed=self.cfg.project.seed,
                num_steps_wait=self.num_steps_wait,
            )
            task_desc = env.task_description
            max_steps = SUITE_MAX_STEPS[suite_name]
            task_successes = 0

            try:
                for episode_idx in range(num_episodes_per_task):
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
                        total_successes += 1
            finally:
                # Always release MuJoCo/EGL resources, even if an episode
                # raises mid-task (otherwise contexts accumulate and crash).
                env.close()

            total_episodes += num_episodes_per_task
            task_success_rate = task_successes / num_episodes_per_task
            per_task_results[task_desc] = {
                "success_rate": task_success_rate,
                "successes": task_successes,
                "episodes": num_episodes_per_task,
            }

        overall_success_rate = (
            total_successes / total_episodes if total_episodes > 0 else 0.0
        )

        return {
            f"eval/success_rate/{suite_name}": overall_success_rate,
            f"eval/per_task/{suite_name}": per_task_results,
            f"eval/total_episodes/{suite_name}": total_episodes,
            f"eval/total_successes/{suite_name}": total_successes,
        }

    def run(self) -> dict[str, Any]:
        """Run rollout evaluation across all configured suites.

        Returns:
            Dict with per-suite success rates, per-task breakdowns,
            and an overall average across suites.
        """
        all_results: dict[str, Any] = {}
        all_suite_rates: list[float] = []

        for suite_name in self.suites:
            logger.info(
                "Evaluating suite %s (%d episodes/task)…",
                suite_name,
                self.num_episodes_per_task,
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
