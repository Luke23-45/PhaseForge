"""State-only robosuite adapter (implementation plan §4.2).

The adapter wraps a robosuite environment behind the exact contracts the
robomimic wrapper used to collect the dataset (``robomimic/envs/robosuite_env.py``
v1.5 track):

* actions are passed **directly** to ``env.step`` — no rescaling. The
  dataset actions are already normalized deltas in ``[-1, 1]``; the
  controller scales them internally via ``input_min/input_max/output_max``.
* state restore mirrors ``reset_to``: ``set_ep_meta`` →
  ``reset_from_xml_string(xml)`` → ``sim.set_state_from_flattened(states)``
  → ``sim.forward()``. A flat ``MjSimState`` is ``[time(1), qpos(nq), qvel(nv)]``.
* the success predicate is the environment's own ``_check_success()``
  (Lift: cube height above table + 0.04).

Failure classification is strict (see :mod:`errors`): the adapter never
swallows simulator errors into task outcomes, and it rejects invalid
policy actions with :class:`PolicyInvalidActionError` before touching the
simulator.

Determinism engineering (2026-08-22 audit). Two classes of hidden state
were measured leaking across episodes in the stock robosuite reset path,
making the "identical resets" protocol promise hold only for
``(time, qpos, qvel)``:

1. Construction-time randomization (e.g. robosuite's ``BoxObject`` sizes
   are drawn from the *global* NumPy RNG inside ``robosuite.make``), so an
   environment's physical geometry depended on the caller's RNG position.
   The adapter therefore constructs the environment under a deterministic
   seed derived from the env name, restoring the caller's RNG state
   afterwards — geometry becomes a pure function of the task, identical
   for every method, seed, and process.
2. Per-episode residue: MuJoCo's ``qacc_warmstart`` solver hint, the OSC
   part controllers' cached references/goals/``initial_joint``, and the
   observables cache all survive ``env.reset()`` and are consumed by the
   first control step of the next episode. ``reset_to`` canonicalizes all
   of them after restoring the serialized state, so every episode of every
   process starts from a bitwise-identical full simulator+controller state
   (proven empirically; see ``tests/evaluations`` regression tests).

``hard_reset=False`` is forced for the same reason: robosuite's default
hard reset re-parses and re-compiles the MJCF model on *every* reset
(~0.54 s/episode measured vs ~2 ms soft) and re-runs construction-time
sampling paths, while the canonicalization above removes everything the
recompile would otherwise have "reset". Physics, model XML, controller
configuration, and the pinned env_kwargs are unchanged.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from phaseforge.evaluations.envs.env_metadata import PinnedEnvMetadata
from phaseforge.evaluations.envs.errors import (
    EnvParityError,
    InfrastructureError,
    PolicyInvalidActionError,
    StateSchemaError,
)

#: Observation keys forced to match the dataset collection pipeline.
#: ``camera_depths`` is passed through because robosuite 1.5 Lift accepts
#: it as a constructor kwarg (the dataset records it in env_kwargs).
#: ``hard_reset=False`` reuses the compiled MJCF model across resets (the
#: default hard reset recompiles it every episode); reset determinism is
#: provided by :meth:`RobosuiteStateAdapter._canonicalize_hidden_state`,
#: not by the recompile.
_FORCED_ENV_KWARGS: dict[str, Any] = {
    "has_renderer": False,
    "has_offscreen_renderer": False,
    "use_camera_obs": False,
    "camera_depths": False,
    "ignore_done": True,
    "use_object_obs": True,
    "hard_reset": False,
}

#: Legacy observation-key alias: robomimic datasets store the object
#: observation under ``object-state`` (classic) or ``object`` (1.5).
_OBS_ALIASES: dict[str, str] = {"object": "object-state"}


@dataclass(frozen=True)
class StateSpec:
    """Declared state schema: ordered ``(key, dim)`` pairs in the contract."""

    keys: tuple[str, ...]
    dims: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.dims):
            raise ValueError(
                f"state_keys and state_dims must have equal length "
                f"(got {len(self.keys)} keys vs {len(self.dims)} dims)"
            )
        if not self.keys:
            raise ValueError("state schema must declare at least one key")
        if any(d <= 0 for d in self.dims):
            raise ValueError("state dims must be positive")

    @property
    def dim(self) -> int:
        return sum(self.dims)

    def index_of(self, key: str) -> tuple[int, int]:
        """Slice ``(start, stop)`` of a key in the concatenated state vector."""
        offset = 0
        for candidate, dim in zip(self.keys, self.dims):
            if candidate == key:
                return offset, offset + dim
            offset += dim
        raise KeyError(f"{key!r} is not part of the state schema {self.keys}")

    @classmethod
    def from_config(cls, state_keys: Any, state_dims: Any) -> StateSpec:
        keys = tuple(str(k) for k in state_keys)
        dims = tuple(int(d) for d in state_dims)
        return cls(keys=keys, dims=dims)


class RobosuiteStateAdapter:
    """Stateless policy-environment adapter for robosuite rollouts.

    The adapter is stateful only about its underlying environment; the
    reset distribution is supplied externally (the frozen reset bank), so
    every episode starts from a pinned simulator state.
    """

    def __init__(
        self,
        meta: PinnedEnvMetadata,
        state_spec: StateSpec,
        *,
        action_dim: int = 7,
        action_low: float = -1.0,
        action_high: float = 1.0,
        horizon: int | None = None,
        seed: int | None = None,
        action_tolerance: float = 1e-4,
    ) -> None:
        self.meta = meta
        self.state_spec = state_spec
        self.action_dim = int(action_dim)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        # Single source of truth for the action-contract tolerance. Both the
        # runner's pre-step check and ``step``'s own guard validate with this
        # value, so a nondefault ``eval.episodes.action_tolerance`` cannot be
        # silently overridden by the adapter's historical hardcoded default
        # (performance review §4, P1).
        self.action_tolerance = float(action_tolerance)
        # The v1.5.1 transport dataset does not serialize ``horizon`` in its
        # env_args, although the evaluation protocol declares a 700-step
        # Transport horizon.  The runner supplies the task-aware fallback;
        # explicit metadata remains authoritative when it is present.
        self._horizon = int(meta.horizon if horizon is None else horizon)
        if self._horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self._horizon}")
        # Keep the protocol seed as adapter metadata, but do not forward it
        # to robosuite.make().  In robosuite 1.5.1 the base MujocoEnv accepts
        # ``seed`` while task classes such as Lift do not expose or forward
        # that argument, so passing it to the registered task raises a
        # TypeError.  Reset-bank generation seeds robosuite's actual global
        # NumPy sampler explicitly instead.
        self.seed = None if seed is None else int(seed)

        try:
            import robosuite
        except ImportError as exc:
            raise EnvParityError(
                "robosuite is not installed. Install the rollout extra "
                "(`uv sync --extra rollout`) — the rollout protocol cannot "
                "run without the simulator."
            ) from exc
        self._robosuite = robosuite

        kwargs = {**meta.env_kwargs, **_FORCED_ENV_KWARGS}
        # Some serialized metadata may carry a generic seed field even
        # though the concrete robosuite task constructor does not accept it.
        # The rollout protocol controls reset determinism in
        # ``generate_reset_bank`` instead of relying on this task-level kwarg.
        kwargs.pop("seed", None)
        self.env = self._make_deterministic(meta.env_name, kwargs)

        try:
            spec = self.env.action_spec
            env_action_dim = int(np.asarray(spec[0]).shape[0])
        except Exception as exc:
            raise EnvParityError(
                f"Could not read the action specification of {meta.env_name!r}"
            ) from exc
        if env_action_dim != self.action_dim:
            raise EnvParityError(
                f"Action dimension mismatch: the environment reports "
                f"{env_action_dim} but the data contract declares "
                f"{self.action_dim}. Refusing to evaluate a different "
                "action space than the dataset."
            )

    def _make_deterministic(self, env_name: str, kwargs: dict[str, Any]) -> Any:
        """Construct the environment under a task-derived deterministic RNG.

        robosuite draws construction-time randomization (object sizes via
        ``mjcf_utils.get_size``, robot init noise) from the *global* NumPy
        RNG inside ``make``. Left alone, an environment's physical geometry
        depends on how many draws the caller made beforehand, so two
        processes (or two methods) could evaluate the same reset bank under
        different cube sizes. Seeding a dedicated deterministic stream
        around ``make`` — derived from the env name only, like the shared
        bank itself — makes geometry a pure function of the task, identical
        across methods, seeds, and processes, without disturbing the
        caller's RNG state.
        """
        digest = hashlib.sha256(f"phaseforge-env-construction::{env_name}".encode())
        construction_seed = int(digest.hexdigest()[:8], 16)
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        np.random.seed(construction_seed)
        random.seed(construction_seed)
        try:
            return self._robosuite.make(env_name, **kwargs)
        except Exception as exc:
            raise EnvParityError(
                f"robosuite.make({env_name!r}) failed with the pinned env_kwargs: {exc}"
            ) from exc
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def extract_state(self, obs: dict[str, Any]) -> np.ndarray:
        """Concatenate the declared state keys from a robosuite obs dict."""
        parts: list[np.ndarray] = []
        for key, dim in zip(self.state_spec.keys, self.state_spec.dims):
            value = obs.get(key)
            if value is None:
                alias = _OBS_ALIASES.get(key)
                if alias is not None:
                    value = obs.get(alias)
            if value is None:
                raise StateSchemaError(
                    f"Observation is missing declared key {key!r} (available: {sorted(obs)})."
                )
            arr = np.asarray(value).reshape(-1)
            if arr.shape[0] != dim:
                raise StateSchemaError(
                    f"Observation key {key!r} has dimension {arr.shape[0]}, declared {dim}."
                )
            parts.append(arr.astype(np.float32))
        state = np.concatenate(parts)
        if not np.isfinite(state).all():
            raise StateSchemaError(
                "Simulator observation contains non-finite values — the "
                "simulation is invalid; this is an infrastructure failure."
            )
        return state

    # ------------------------------------------------------------------
    # Reset (mirrors robomimic reset_to)
    # ------------------------------------------------------------------

    def reset_to(
        self,
        states: np.ndarray,
        *,
        xml: str | None = None,
        ep_meta: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Restore a pinned simulator state and return the initial obs state.

        Mirrors ``EnvRobosuite.reset_to`` step-for-step so the reset
        distribution is exactly the dataset's, then canonicalizes the
        hidden simulator/controller state that the serialized flat state
        does not carry (see the module docstring): without this, the first
        control step of an episode is polluted by the previous episode's
        solver warm-start, OSC cached references/goals, ``initial_joint``,
        and observable caches, making episodes order-dependent.
        """
        try:
            if ep_meta is not None and hasattr(self.env, "set_ep_meta"):
                self.env.set_ep_meta(ep_meta)
            if xml is not None:
                self.env.reset_from_xml_string(xml)
            else:
                self.env.reset()
            self.env.sim.set_state_from_flattened(np.asarray(states, dtype=np.float64))
            self._canonicalize_hidden_state()
        except InfrastructureError:
            raise
        except Exception as exc:
            raise InfrastructureError(
                f"State restore failed for reset case (xml={xml is not None}): {exc}"
            ) from exc
        try:
            obs = self.env._get_observations(force_update=True)
        except Exception as exc:
            raise InfrastructureError(
                f"Observation extraction failed after state restore: {exc}"
            ) from exc
        return self.extract_state(obs)

    def _canonicalize_hidden_state(self) -> None:
        """Erase episode-order dependence after a state restore.

        ``sim.set_state_from_flattened`` writes only ``(time, qpos, qvel)``.
        Everything else in the simulator and the robosuite controller stack
        keeps whatever the previous episode left there and is consumed by
        the next first control step:

        * ``qacc_warmstart`` — MuJoCo's solver warm-start hint (measured at
          O(10) after an episode vs 0 for a fresh sim);
        * the OSC part controllers' ``ref_pos``/``ref_ori_mat``/``J_*``/
          ``mass_matrix``/``joint_pos`` caches, their ``goal_pos``/
          ``goal_ori``, and ``initial_joint`` (the reset-time joint sample
          used for nullspace actions);
        * the observables cache (refreshed by the ``force_update`` poll in
          :meth:`reset_to` right after this call).

        For each part controller, ``update(force=True)`` first refreshes
        every cached quantity (``joint_pos`` included) from the restored
        simulator state, then ``update_initial_joints(joint_pos)`` stores
        those *restored* joints as the initial configuration (and, for OSC
        controllers, also resets the goal to the achieved pose). The order
        matters: reading ``joint_pos`` before the forced update would
        canonicalize to the previous episode's joints.
        """
        sim = self.env.sim
        sim.data.qacc_warmstart[:] = 0.0
        sim.forward()
        for robot in getattr(self.env, "robots", []):
            composite = getattr(robot, "composite_controller", None)
            parts = getattr(composite, "part_controllers", None)
            if not isinstance(parts, dict):
                continue
            for part_controller in parts.values():
                update = getattr(part_controller, "update", None)
                if callable(update):
                    update(force=True)
                update_initial = getattr(part_controller, "update_initial_joints", None)
                joint_pos = getattr(part_controller, "joint_pos", None)
                if callable(update_initial) and joint_pos is not None:
                    update_initial(np.asarray(joint_pos))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def validate_action(
        self, action: np.ndarray, *, tolerance: float | None = None
    ) -> np.ndarray:
        """Validate a policy action against the data contract; return float64.

        ``tolerance=None`` resolves to the adapter's configured
        ``action_tolerance`` (from ``eval.episodes.action_tolerance``), so
        every validation site enforces the same contract.

        Raises:
            PolicyInvalidActionError: NaN, non-finite, wrong shape, or out
                of the declared ``[action_low, action_high]`` range (with
                ``tolerance``).
        """
        if tolerance is None:
            tolerance = self.action_tolerance
        arr = np.asarray(action)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr.reshape(-1)
        if arr.shape != (self.action_dim,):
            raise PolicyInvalidActionError(
                f"Action has shape {arr.shape}, expected ({self.action_dim},)"
            )
        if not np.isfinite(arr).all():
            bad = int(np.count_nonzero(~np.isfinite(arr)))
            raise PolicyInvalidActionError(f"Action contains {bad} non-finite value(s) (NaN/Inf).")
        low = self.action_low - tolerance
        high = self.action_high + tolerance
        if float(arr.min()) < low or float(arr.max()) > high:
            raise PolicyInvalidActionError(
                f"Action outside the declared contract "
                f"[{self.action_low}, {self.action_high}] (+{tolerance} tol): "
                f"min={float(arr.min()):.4g}, max={float(arr.max()):.4g}."
            )
        return arr.astype(np.float64)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool, dict[str, Any]]:
        """Execute one policy action; returns ``(state, done, success, info)``.

        ``done`` is always False by construction (``ignore_done=True`` —
        horizon termination is the runner's responsibility). ``success`` is
        the environment's own predicate. Simulator exceptions are wrapped
        as :class:`InfrastructureError`, never as task outcomes.
        """
        validated = self.validate_action(action)
        try:
            obs, _reward, _done, _info = self.env.step(validated)
        except Exception as exc:
            raise InfrastructureError(
                f"Simulator step failed: {type(exc).__name__}: {exc}"
            ) from exc
        state = self.extract_state(obs)
        success = self.check_success()
        return state, False, success, {}

    def check_success(self) -> bool:
        """The environment's own success predicate (Lift: ``task`` key)."""
        try:
            result = self.env._check_success()
        except Exception as exc:
            raise InfrastructureError(
                f"Success predicate failed: {type(exc).__name__}: {exc}"
            ) from exc
        if isinstance(result, dict):
            task_success = result.get("task")
            if task_success is None:
                raise InfrastructureError(
                    f"Success predicate dict has no 'task' key: {sorted(result)}"
                )
            return bool(task_success)
        return bool(result)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @property
    def horizon(self) -> int:
        return self._horizon

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


__all__ = [
    "StateSpec",
    "RobosuiteStateAdapter",
    "_FORCED_ENV_KWARGS",
]
