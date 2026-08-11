"""State-only LIBERO environment wrapper.

Wraps ``OffScreenRenderEnv`` to produce the state vector that matches the
training data format: the 23-DoF proprioceptive block (7 joint pos + 7
joint vel + 3 EEF pos + 4 EEF quat + 2 gripper qpos), optionally extended
with the P-Stage 1 object-state channel (per-object world pos + quat +
occupancy mask, in the same layout the ingest pipeline decodes from the
mirror's ``states`` arrays). All image observations are discarded.
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


#: The exact observable names that build the 23-DoF proprioceptive block
#: (``_extract_state``). Everything else — camera images AND object sensors
#: that are NOT part of the configured object-state channel — is unused and
#: is disabled to eliminate per-step rendering / sensor-update costs.
KEPT_OBSERVABLE_NAMES: tuple[str, ...] = (
    "robot0_joint_pos",     # 7
    "robot0_joint_vel",     # 7
    "robot0_eef_pos",       # 3
    "robot0_eef_quat",      # 4
    "robot0_gripper_qpos",  # 2
)


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


def _disable_unused_observables(env: Any, keep_extra: tuple[str, ...] = ()) -> None:
    """Keep only the observables that build the state vector.

    The policy is state-only; robosuite would otherwise re-render camera
    images and re-query every object sensor on each ``step()`` (the
    dominant per-step costs — see robosuite issue #722). Only the
    proprioceptive observables in :data:`KEPT_OBSERVABLE_NAMES` plus the
    per-object ``{name}_pos`` / ``{name}_quat`` observables listed in
    ``keep_extra`` are kept, so ``_extract_state`` is unaffected.

    Always logs a diagnostic (kept/disabled counts) so a silent no-op —
    e.g. a robosuite API difference — is visible in the eval log instead
    of quietly leaving the sensors on.
    """
    target_env, observables = _find_observable_env(env)
    if not observables:
        logger.warning(
            "Observable pruning NO-OP: no _observables found on %s "
            "(or any wrapped _env/env) — rendering and object sensors "
            "will keep running every step.",
            type(env).__name__,
        )
        return

    keep = set(KEPT_OBSERVABLE_NAMES) | set(keep_extra)
    disabled = 0
    kept = 0
    for name, obs in observables.items():
        if name in keep:
            kept += 1
            continue
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
                "Observable %r has no set_enabled/set_active — cannot prune it.",
                name,
            )
        except Exception:
            logger.warning(
                "Could not disable observable %r — it stays enabled.",
                name,
                exc_info=True,
            )

    logger.info(
        "Observable pruning: kept %d/%d (state vector), disabled %d "
        "— no per-step rendering/sensor updates for the rest.",
        kept, len(observables), disabled,
    )


class StateOnlyLiberoEnv:
    """State-only wrapper around LIBERO's OffScreenRenderEnv.

    Constructs the state vector that matches our training data format:
    the 23-DoF proprioceptive block plus, when ``object_state_cfg`` is
    provided, the per-task object-state channel (same object selection and
    ordering as the ingest-side decode — both read the census-built
    :class:`ObjectIndex`). All image observations are discarded (and, by
    default, not even rendered).

    Usage:
        env = StateOnlyLiberoEnv(suite_name="libero_spatial", task_id=0, seed=42)
        state = env.reset(episode_idx=0)       # (state_dim,) numpy array
        next_state, reward, terminated, truncated, info = env.step(action)
        env.close()

    Note: Gymnasium-style step() return: (obs, reward, terminated, truncated, info).

    ``hard_reset`` mirrors LIBERO/robosuite: ``True`` (default) rebuilds the
    MuJoCo sim from XML on every ``reset()`` — the official benchmark
    behavior, bit-identical across runs. ``False`` reuses the existing sim
    (faster resets) but is NOT bit-identical after settling steps
    (LeRobot docs, "Reset performance").
    """

    def __init__(
        self,
        suite_name: str,
        task_id: int,
        seed: int = 42,
        num_steps_wait: int = 10,
        render_observations: bool = False,
        hard_reset: bool = True,
        object_state_cfg: Any = None,
    ) -> None:
        # Prevent LIBERO interactive prompt hanging in non-interactive environments
        libero_cfg = Path.home() / ".libero" / "config.yaml"
        if not libero_cfg.exists():
            libero_cfg.parent.mkdir(parents=True, exist_ok=True)
            libero_cfg.write_text("DATASET_PATH: ''\n")

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

        # P-Stage 1 object-state channel: same census-built index as the
        # ingest side, so train <-> eval object selection matches by
        # construction (E3).
        self._object_names: list[str] = []
        self._object_k_slots = 0
        self._object_dim = 7
        self._object_include_mask = True
        if object_state_cfg is not None:
            from omegaconf import DictConfig, OmegaConf

            oscfg = (
                object_state_cfg
                if isinstance(object_state_cfg, DictConfig)
                else OmegaConf.create(dict(object_state_cfg))
            )
            if oscfg.get("enabled", True):
                from phaseforge.data.libero.object_state import ObjectIndex
                from phaseforge.data.paths import resolve_object_index_path

                # Same resolver as the cache identity, ingest FSM and
                # manifest provenance — one source of truth for the index
                # location, so train and eval can never drift apart.
                path = resolve_object_index_path(
                    OmegaConf.create({"object_state": oscfg})
                )
                object_index = ObjectIndex.load(path)
                self._object_index = object_index
                self._object_names = object_index.object_names(self._task.name)
                self._object_k_slots = object_index.k_slots
                self._object_dim = object_index.dim_per_object
                self._object_include_mask = object_index.include_mask
                logger.info(
                    "Object-state channel enabled for %s: %d object(s) "
                    "(k_slots=%d, dim_per_object=%d)",
                    self._task.name, len(self._object_names),
                    self._object_k_slots, self._object_dim,
                )

        bddl_file = str(
            Path(get_libero_path("bddl_files"))
            / self._task.problem_folder
            / self._task.bddl_file
        )
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=128,
            camera_widths=128,
            hard_reset=hard_reset,
        )
        if not render_observations:
            keep_extra = tuple(
                name
                for obj in self._object_names
                for name in (f"{obj}_pos", f"{obj}_quat")
            )
            _disable_unused_observables(self._env, keep_extra=keep_extra)

        # LIBERO's BDDLBaseDomain.reward() re-evaluates the full BDDL
        # success predicate on every step (bddl_base_domain.py:167-191) even
        # though we discard the reward: ``done`` from step()
        # (bddl_base_domain.py:809) is the very same predicate result. Stub
        # reward() so each control step runs the predicate exactly once
        # (via done) instead of twice. The stub is guarded: if the attribute
        # chain ever changes, we degrade to the old behavior with a warning.
        underlying = getattr(self._env, "env", None)
        if underlying is not None and hasattr(underlying, "reward"):
            underlying.reward = lambda action: 0.0
        else:
            logger.warning(
                "Reward stub skipped: no underlying env.reward found on %s "
                "— _check_success() still runs twice per step.",
                type(self._env).__name__,
            )

        # Joint-velocity finite-difference history (parity contract, E3):
        # None means "no previous position" -> t=0 velocity is zeros,
        # exactly like the ingest-side np.diff(prepend=first_row).
        self._prev_joint_pos: np.ndarray | None = None

    @property
    def task_description(self) -> str:
        """Human-readable task description from the benchmark."""
        return self._task.language

    @property
    def num_init_states(self) -> int:
        """Number of available initial-state configurations."""
        return len(self._init_states)

    #: Joint-velocity convention (parity contract, E3): the ingest side
    #: derives ``robot0_joint_vel`` as the finite difference of joint
    #: positions with ``prepend=first_row`` (vision_stripper.py). The
    #: simulator's raw ``qvel`` is a physical rad/s velocity and would NOT
    #: match that per-timestep delta, silently shifting 7 of 23 proprio
    #: dims between training and rollout. To guarantee train <-> eval
    #: parity, the wrapper therefore OVERWRITES the observed joint velocity
    #: with the same finite difference before building the state vector
    #: (first frame -> zeros, exactly like ``prepend`` at t=0).
    _PREV_JOINT_POS_ATTR = "_prev_joint_pos"

    def _apply_joint_vel_fd(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Return the joint-vel block as ``pos[t] - pos[t-1]`` (first = 0).

        Reads ``robot0_joint_pos``, compares against the previously
        returned joint position and stores the current one for the next
        call. ``None`` previous (i.e. right after reset) yields zeros,
        mirroring the training-side ``prepend`` convention.
        """
        pos = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
        prev = getattr(self, self._PREV_JOINT_POS_ATTR, None)
        vel = np.zeros(pos.shape, dtype=np.float32) if prev is None else pos - prev
        setattr(self, self._PREV_JOINT_POS_ATTR, pos)
        return vel

    def _extract_state(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate the state vector from the observation dict.

        Layout (identical to the ingest side):
            [ proprio (23) | object_block (k_slots*7) | mask (k_slots) ]

        Object poses come from the ``{name}_pos`` / ``{name}_quat``
        observables (kept enabled for this task's objects); empty slots are
        zero-padded and the occupancy mask is appended. The joint-velocity
        block is the finite-difference convention of the ingest side (see
        :meth:`_apply_joint_vel_fd`) — never the raw simulator ``qvel``.
        """
        parts = [
            obs["robot0_joint_pos"],    # 7
            self._apply_joint_vel_fd(obs),  # 7 — finite-diff, matches ingest
            obs["robot0_eef_pos"],      # 3
            obs["robot0_eef_quat"],     # 4
            obs["robot0_gripper_qpos"], # 2
        ]
        n_objects = len(self._object_names)
        if n_objects:
            from phaseforge.data.libero.object_state import robosuite_quat_to_wxyz

            for name in self._object_names:
                parts.append(obs[f"{name}_pos"])
                # Robosuite emits object quaternions as xyzw; the training
                # HDF5 qpos/object decoder state contract is wxyz.
                parts.append(
                    robosuite_quat_to_wxyz(obs[f"{name}_quat"])
                )
            pad = self._object_k_slots - n_objects
            if pad > 0:
                parts.append(np.zeros(pad * self._object_dim, dtype=np.float32))
            if self._object_include_mask:
                mask = np.zeros(self._object_k_slots, dtype=np.float32)
                mask[:n_objects] = 1.0
                parts.append(mask)
        return np.concatenate(parts).astype(np.float32)

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
        # Parity contract: joint velocity is a finite difference, so the
        # "previous" position must not leak across episodes (t=0 -> zeros,
        # exactly like the ingest-side `prepend`).
        setattr(self, self._PREV_JOINT_POS_ATTR, None)
        return self._extract_state(obs)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take a step and return Gymnasium-style (obs, reward, terminated, truncated, info).

        ``terminated = done or is_success`` and ``truncated = False`` so
        that any episode end is a termination (the caller distinguishes
        success vs. timeout via ``info["is_success"]``).

        Success is detected with the EXPLICIT ``env.check_success()`` call
        on every step (the official LeRobot LIBERO wrapper pattern) — never
        inferred from ``done`` alone. ``done`` can be raised by horizon
        exhaustion or by the BDDL predicate; trusting it after a one-step
        probe is unsafe (a probe result does not hold forever, and a
        horizon termination must never be miscounted as success).
        ``is_success`` is therefore always the explicit predicate result,
        and ``terminated`` combines it with the environment's ``done``.

        The reward() stub installed at construction keeps the BDDL
        predicate evaluation at one call per step (robosuite's reward()
        would otherwise re-evaluate it); the explicit check_success()
        below is the authoritative source for ``is_success``.
        """
        action = np.asarray(action, dtype=np.float32).flatten()
        obs, reward, done, info = self._env.step(action)
        self._elapsed_steps += 1
        # Explicit success check every step — official wrapper semantics
        # (LeRobot libero.py), no inference from `done`.
        is_success = bool(self._env.check_success())
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
