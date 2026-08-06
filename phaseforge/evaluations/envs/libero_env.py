"""State-only LIBERO environment wrapper.

Wraps ``OffScreenRenderEnv`` to produce the 23-DoF state vector
(7 joint pos + 7 joint vel + 3 EEF pos + 4 EEF quat + 2 gripper qpos)
that matches the training data format. All image observations are discarded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

SUITE_BENCHMARK_NAMES: dict[str, str] = {
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
    "libero_long": "libero_10",
    "libero_10": "libero_10",
    "libero_90": "libero_90",
}


def _find_observable_env(
    env: Any,
) -> tuple[Any | None, dict[str, Any] | None]:
    """Locate the robosuite env that owns the observable dict.

    LIBERO's ``OffScreenRenderEnv`` is a composition wrapper: the real
    robosuite env — and its ``_observables`` dict — lives at ``env._env``
    (confirmed on Colab: the wrapper itself has no ``_observables``).
    Walks nested ``_env``/``env`` attributes defensively and guards
    against cycles.
    """
    seen: set[int] = set()
    current: Any = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        observables = getattr(current, "_observables", None)
        if observables:
            return current, observables
        nested = getattr(current, "_env", None)
        if nested is None:
            nested = getattr(current, "env", None)
        current = nested
    return None, None


def _disable_image_observables(env: Any) -> None:
    """Disable image-modality observables so no per-step rendering occurs.

    The policy is state-only; robosuite would otherwise re-render camera
    images on every ``step()`` (the dominant evaluation cost — see
    robosuite issue #722). State-modality observables (joint pos/vel, EEF,
    gripper) are untouched, so the 23-DoF state vector is unaffected.

    Always logs a diagnostic (observable count, modalities, disabled
    count) so a silent no-op — e.g. a robosuite API difference — is
    visible in the eval log instead of quietly leaving rendering on.
    """
    target_env, observables = _find_observable_env(env)
    if not observables:
        logger.warning(
            "Rendering skip NO-OP: no _observables found on %s "
            "(or any wrapped _env/env) — image observables will keep "
            "rendering every step.",
            type(env).__name__,
        )
        return

    modalities: dict[str, int] = {}
    disabled = 0
    image_obs = 0
    for name, obs in observables.items():
        modality = getattr(obs, "modality", None)
        modalities[str(modality)] = modalities.get(str(modality), 0) + 1
        if modality != "image":
            continue
        image_obs += 1
        try:
            if hasattr(obs, "set_enabled"):
                obs.set_enabled(False)
                disabled += 1
                continue
            if hasattr(obs, "set_active"):  # robosuite < 1.4 fallback
                obs.set_active(False)
                disabled += 1
                continue
            logger.warning(
                "Observable %r is image-modality but has no set_enabled/set_active.",
                name,
            )
        except Exception:
            logger.warning(
                "Could not disable image observable %r — rendering stays enabled.",
                name,
                exc_info=True,
            )

    logger.info(
        "Rendering skip: disabled %d/%d image observable(s) "
        "(total observables: %d, modalities: %s)",
        disabled, image_obs, len(observables), modalities,
    )


class StateOnlyLiberoEnv:
    """State-only wrapper around LIBERO's OffScreenRenderEnv.

    Constructs the 23-DoF state vector that matches our training data format.
    All image observations are discarded (and, by default, not even rendered).

    Usage:
        env = StateOnlyLiberoEnv(suite_name="libero_spatial", task_id=0, seed=42)
        state = env.reset(episode_idx=0)       # (23,) numpy array
        next_state, reward, terminated, truncated, info = env.step(action)
        env.close()

    Note: Gymnasium-style step() return: (obs, reward, terminated, truncated, info).
    """

    def __init__(
        self,
        suite_name: str,
        task_id: int,
        seed: int = 42,
        num_steps_wait: int = 10,
        render_observations: bool = False,
    ) -> None:
        try:
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as exc:
            raise RuntimeError(
                "libero/robosuite packages not installed. "
                "Run: uv add robosuite && pip install libero"
            ) from exc

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

        bddl_file = str(
            Path(get_libero_path("bddl_files"))
            / self._task.problem_folder
            / self._task.bddl_file
        )
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=128,
            camera_widths=128,
        )
        if not render_observations:
            _disable_image_observables(self._env)

    @property
    def task_description(self) -> str:
        """Human-readable task description from the benchmark."""
        return self._task.language

    @property
    def num_init_states(self) -> int:
        """Number of available initial-state configurations."""
        return len(self._init_states)

    def _extract_state(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate the 23-DoF state vector from the observation dict."""
        return np.concatenate(
            [
                obs["robot0_joint_pos"],    # 7
                obs["robot0_joint_vel"],    # 7
                obs["robot0_eef_pos"],      # 3
                obs["robot0_eef_quat"],     # 4
                obs["robot0_gripper_qpos"], # 2
            ]
        ).astype(np.float32)

    def reset(self, episode_idx: int = 0) -> np.ndarray:
        """Reset env to the given initial state and return the initial state vector.

        Follows the exact LIBERO reset sequence:
        1. env.seed(seed)
        2. env.reset()
        3. env.set_init_state(init_states[episode_idx])
        4. num_steps_wait dummy actions to let objects settle
        """
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

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take a step and return Gymnasium-style (obs, reward, terminated, truncated, info).

        Maps robosuite's ``done`` (horizon reached) + ``check_success()``
        (goal predicates satisfied).  Following the LeRobot convention,
        ``terminated = done or is_success`` and ``truncated = False`` so
        that any episode end is a termination (the caller distinguishes
        success vs. timeout via ``info["is_success"]``).
        """
        action = np.asarray(action, dtype=np.float32).flatten()
        obs, reward, done, info = self._env.step(action)
        self._elapsed_steps += 1
        is_success = self._env.check_success()
        terminated = bool(done) or bool(is_success)
        truncated = False
        state = self._extract_state(obs)
        return (
            state,
            float(reward),
            terminated,
            truncated,
            {"is_success": is_success, "elapsed_steps": self._elapsed_steps},
        )

    def close(self) -> None:
        """Clean up the underlying environment."""
        if self._env is not None:
            self._env.close()
