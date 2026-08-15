"""Scripted state-oracle controllers for the five robomimic benchmark tasks.

A training-free deterministic controller used to validate the rollout
pipeline end-to-end (plan §4.5, gate 4): if the per-task controller cannot
solve its frozen reset bank (default threshold: all cases), the adapter /
bank / predicate chain is broken and the real policy rollouts would be
meaningless.

The :class:`ScriptedController` is the base phase machine. It emits
normalized OSC delta actions exactly like the dataset's:
``desired_delta / output_max`` with ``output_max = (0.05, 0.05, 0.05, 0.5,
0.5, 0.5)`` (Panda default, matched by the robomimic action_scale
contract). Subclasses override three hooks:

* ``object_pos(state)`` -- 3D point to approach and grasp.
* ``placement_target(state)`` -- 3D point to release above. ``None`` means
  the task has no separate placement phase (e.g. Lift only needs to lift).
* ``is_success(state)`` -- environment's success predicate (mirrored from
  ``env._check_success()``).

The base state schema (5-task protocol, single-arm Panda OSC_POSE) is
``robot0_eef_pos(3) robot0_eef_quat(4) robot0_gripper_qpos(2) object(10)``
where ``object[0:3]`` is the manipulated object's center. The
:class:`~phaseforge.evaluations.envs.task_registry.TaskSpec` for each
task carries the canonical keys/dims; this file is concerned only with
the policy, not the schema declaration.

Phase progression for tasks with a placement phase (Can, Square, ToolHang):

    APPROACH -> DESCEND -> GRASP_HOLD -> LIFT -> TRANSPORT -> PLACE

Phase progression for tasks without placement (Lift):

    APPROACH -> DESCEND -> GRASP_HOLD -> LIFT

A stall watchdog abandons an episode when the end-effector makes no
progress while tracking a target. The GRASP phase is exempt (closing
the gripper intentionally keeps the end-effector stationary). The
abandoned episode then simply times out as a task failure, never as a
policy / simulator error.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from phaseforge.evaluations.envs.robosuite_adapter import StateSpec

#: Default Panda OSC output limits (matching the dataset's action contract).
POSITION_SCALE: float = 0.05
ORIENTATION_SCALE: float = 0.5

#: Lift geometry: table top at 0.8; success = cube z > 0.84.
TABLE_HEIGHT: float = 0.8
SUCCESS_Z: float = TABLE_HEIGHT + 0.04
OBJECT_HALF_SIZE: float = 0.0215

#: Gripper action convention: -1 closes, +1 opens (dataset convention).
GRIPPER_OPEN: float = 1.0
GRIPPER_CLOSE: float = -1.0

#: Steps to hold the gripper closed before lifting.
GRASP_HOLD_STEPS: int = 25

#: Steps to hold the gripper open above the placement target before releasing.
PLACE_HOLD_STEPS: int = 10

#: If the end-effector makes no progress within this many steps while
#: tracking a target, the episode is abandoned (stays in the final phase
#: with zero translation commands until the horizon expires -- recorded as
#: a task timeout, never as a simulator/policy failure).
STALL_STEPS: int = 30
STALL_PROGRESS: float = 0.005

#: Default placement target height (relative to the table) for tasks that
#: transport an object to a separate receptacle.
DEFAULT_PLACEMENT_Z: float = 0.95


class _Phase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    TRANSPORT = auto()
    PLACE = auto()
    STALLED = auto()


@dataclass(frozen=True)
class ScriptedControllerConfig:
    """Tunables for the scripted controllers (defaults = validated values)."""

    position_scale: float = POSITION_SCALE
    orientation_scale: float = ORIENTATION_SCALE
    success_z: float = SUCCESS_Z
    descend_z_offset: float = OBJECT_HALF_SIZE + 0.02
    approach_z_offset: float = 0.12
    lift_z: float = 0.95
    placement_z: float = DEFAULT_PLACEMENT_Z
    grasp_hold_steps: int = GRASP_HOLD_STEPS
    place_hold_steps: int = PLACE_HOLD_STEPS
    stall_steps: int = STALL_STEPS
    stall_progress: float = STALL_PROGRESS
    position_tolerance: float = 0.01


class ScriptedController:
    """Base phase-machine controller emitting normalized OSC delta actions.

    Subclasses override :meth:`object_pos`, :meth:`placement_target`, and
    :meth:`is_success` to specialize for a specific task. Tasks that do
    not require a separate placement step (Lift) leave
    :meth:`placement_target` returning ``None``; the phase machine then
    stops at :attr:`_Phase.LIFT` and waits for the success predicate.
    """

    object_key: str = "object"
    """State-key carrying the manipulated object's pose (default: ``"object"``)."""

    eef_key: str = "robot0_eef_pos"
    """State-key carrying the end-effector position (default: ``"robot0_eef_pos"``)."""

    def __init__(
        self,
        state_spec: StateSpec,
        *,
        config: ScriptedControllerConfig | None = None,
    ) -> None:
        self.state_spec = state_spec
        self.eef_start, self.eef_end = state_spec.index_of(self.eef_key)
        self.obj_start, self.obj_end = state_spec.index_of(self.object_key)
        # Transport has a second arm and therefore a 14-dimensional action
        # space. The controller currently drives robot0 only; robot1 receives
        # a neutral/open command. This is a dimensional adapter, not a claim
        # that the resulting controller solves the two-arm task.
        self._two_arm = "robot1_eef_pos" in state_spec.keys
        self._object_is_indicator = self.obj_end - self.obj_start < 3
        self.config = config or ScriptedControllerConfig()
        if self.config.position_scale <= 0:
            raise ValueError("position_scale must be positive")
        self.reset()

    def reset(self) -> None:
        """Clear per-episode phase memory (call at episode start)."""
        self._phase = _Phase.APPROACH
        self._approach_done = False
        self._grasp_started_at: int | None = None
        self._place_started_at: int | None = None
        self._stall_since: int | None = None
        self._last_eef: np.ndarray | None = None
        self._placement_snapshot: np.ndarray | None = None

    # ------------------------------------------------------------------
    # State parsing
    # ------------------------------------------------------------------

    def eef_pos(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state[self.eef_start : self.eef_end], dtype=np.float64)

    def object_pos(self, state: np.ndarray) -> np.ndarray:
        """3D position of the manipulated object.

        Override for tasks where the object is not at the start of the
        ``object`` key (e.g. a different state's offset convention).

        If a declared object key carries fewer than 3 dimensions, the
        controller falls back to the end-effector as a stand-in. This is only
        a defensive compatibility path; it is not a task-success oracle.
        """
        if self._object_is_indicator:
            return self.eef_pos(state).copy()
        return np.asarray(state[self.obj_start : self.obj_start + 3], dtype=np.float64)

    def placement_target(self, state: np.ndarray) -> np.ndarray | None:
        """3D target above the receptacle; ``None`` if no placement phase.

        Default: no placement phase. Subclasses for tasks with a separate
        receptacle (Can, Square, ToolHang) override this to point to the
        receptacle's ``(x, y)`` with ``z = placement_z``.
        """
        return None

    def is_success(self, state: np.ndarray) -> bool:
        """Mirror of the environment's success predicate.

        Default: object z above the table by the success margin (Lift).
        Subclasses override for tasks with a different predicate.
        """
        return bool(self.object_pos(state)[2] > self.config.success_z)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray, t: int) -> np.ndarray:
        """Return a normalized action ``(7,)`` for step ``t`` of an episode."""
        config = self.config
        eef = self.eef_pos(state)
        obj = self.object_pos(state)

        if self.is_success(state):
            return self._hold_action(GRIPPER_OPEN)

        self._watchdog(eef, t)

        if self._phase is _Phase.STALLED:
            return self._hold_action(GRIPPER_OPEN)

        if self._phase is _Phase.APPROACH and not self._approach_done:
            target = obj + np.array([0.0, 0.0, config.approach_z_offset])
            if np.linalg.norm(target - eef) < config.position_tolerance:
                self._approach_done = True
                self._phase = _Phase.DESCEND
            else:
                return self._track(target, GRIPPER_OPEN, eef)

        if self._phase is _Phase.DESCEND:
            target = obj + np.array([0.0, 0.0, config.descend_z_offset])
            if np.linalg.norm(target - eef) < config.position_tolerance:
                self._phase = _Phase.GRASP
                self._grasp_started_at = t
            else:
                return self._track(target, GRIPPER_OPEN, eef)

        if self._phase is _Phase.GRASP:
            assert self._grasp_started_at is not None
            if t - self._grasp_started_at >= config.grasp_hold_steps:
                # Snapshot the placement target at the moment the grasp
                # completes. The target must be relative to the object's
                # position *at the time of the grasp*, not the current
                # (already grasped, already moving) object position; the
                # latter would chase the eef indefinitely.
                self._snapshot_placement_target(state)
                self._phase = _Phase.LIFT
            else:
                return self._hold_action(GRIPPER_CLOSE)

        placement = self._placement_target()
        has_placement = placement is not None

        if self._phase is _Phase.LIFT:
            target = np.array([obj[0], obj[1], config.lift_z])
            # LIFT is "done" once the eef reaches or exceeds the lift
            # height (within one step of the discrete step size).
            # The xy axis is already aligned with the object by then,
            # so we only need to confirm the eef is high enough.
            if eef[2] >= config.lift_z - config.position_scale:
                if has_placement:
                    self._phase = _Phase.TRANSPORT
                # else (no placement): stay in LIFT and let the success
                # predicate fire; the track action is now near-zero
                # because we are at the target.
            return self._track(target, GRIPPER_CLOSE, eef)

        if self._phase is _Phase.TRANSPORT and has_placement and placement is not None:
            assert placement is not None
            if (
                abs(placement[0] - eef[0]) < config.position_scale
                and abs(placement[1] - eef[1]) < config.position_scale
                and abs(placement[2] - eef[2]) < config.position_scale
            ):
                self._phase = _Phase.PLACE
                self._place_started_at = t
            else:
                return self._track(placement, GRIPPER_CLOSE, eef)

        if self._phase is _Phase.PLACE and placement is not None:
            assert self._place_started_at is not None
            if t - self._place_started_at >= config.place_hold_steps:
                return self._hold_action(GRIPPER_OPEN)
            return self._track(placement, GRIPPER_OPEN, eef)

        return self._hold_action(GRIPPER_OPEN)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _track(self, target: np.ndarray, gripper: float, eef: np.ndarray) -> np.ndarray:
        delta = target - eef
        return self._normalized_action(delta, gripper)

    def _snapshot_placement_target(self, state: np.ndarray) -> None:
        """Pin the placement target to ``self._placement_snapshot``.

        Called at the moment the grasp completes. Subclasses compute the
        target via :meth:`placement_target`; the base class records it so
        the TRANSPORT phase tracks a fixed receptacle even while the
        object follows the moving end-effector.
        """
        target = self.placement_target(state)
        if target is not None:
            self._placement_snapshot = np.asarray(target, dtype=np.float64)

    def _placement_target(self) -> np.ndarray | None:
        """The pinned placement target recorded at grasp completion.

        Falls back to the live (dynamic) :meth:`placement_target` if no
        snapshot exists yet, preserving the no-grasp controller path.
        """
        if self._placement_snapshot is not None:
            return self._placement_snapshot
        return self.placement_target(np.zeros(self.state_spec.dim, dtype=np.float64))

    def _hold_action(self, gripper: float) -> np.ndarray:
        action = np.zeros(14 if self._two_arm else 7, dtype=np.float64)
        action[6] = gripper
        if self._two_arm:
            action[13] = GRIPPER_OPEN
        return action

    def _normalized_action(self, delta: np.ndarray, gripper: float) -> np.ndarray:
        config = self.config
        action = np.zeros(14 if self._two_arm else 7, dtype=np.float64)
        action[0:3] = np.clip(delta / config.position_scale, -1.0, 1.0)
        action[6] = gripper
        if self._two_arm:
            action[13] = GRIPPER_OPEN
        return action

    def _watchdog(self, eef: np.ndarray, t: int) -> None:
        """Abandon an episode when tracking makes no progress.

        The GRASP phase is exempt: closing the gripper intentionally keeps
        the end-effector stationary, so a lack of movement there is
        expected, not a stall.
        """
        if self._phase is _Phase.GRASP:
            return
        config = self.config
        if self._last_eef is None:
            self._last_eef = eef.copy()
            self._stall_since = t
            return
        moved = np.linalg.norm(eef - self._last_eef)
        self._last_eef = eef.copy()
        if moved < config.stall_progress:
            if self._stall_since is None:
                self._stall_since = t
            elif t - self._stall_since >= config.stall_steps:
                self._phase = _Phase.STALLED
        else:
            self._stall_since = None


class ScriptedLiftController(ScriptedController):
    """Scripted oracle for the Lift task: pick up the cube and lift it.

    Success predicate: object z > 0.84 (above table by 4 cm).
    No placement phase -- the task is complete once lifted.
    """

    object_key = "object"

    def placement_target(self, state: np.ndarray) -> np.ndarray | None:  # noqa: ARG002
        return None


class ScriptedCanController(ScriptedController):
    """Scripted oracle for the Can task: pick up the can, place it in the bin.

    Success predicate (env._check_success, mirrored): can inside the bin
    (absolute xy) AND lifted above the table. The receptacle location is
    absolute (not relative to the object) so it matches the kinematic
    fake sim's success criterion and the real env._check_success().

    Placement target: absolute receptacle xy + placement_z.
    """

    #: Absolute receptacle xy for the Can task (mirrors the fake sim).
    RECEPTACLE_XY: tuple[float, float] = (0.15, 0.15)

    object_key = "object"

    def placement_target(self, state: np.ndarray) -> np.ndarray | None:  # noqa: ARG002
        config = self.config
        return np.array([self.RECEPTACLE_XY[0], self.RECEPTACLE_XY[1], config.placement_z])

    def is_success(self, state: np.ndarray) -> bool:
        """Mirror of env._check_success: can in bin AND lifted."""
        obj = self.object_pos(state)
        if obj[2] <= self.config.success_z:
            return False
        placement = self._placement_target()
        assert placement is not None
        return bool(
            abs(obj[0] - placement[0]) < self.config.position_scale
            and abs(obj[1] - placement[1]) < self.config.position_scale
        )


class ScriptedSquareController(ScriptedController):
    """Scripted oracle for the NutAssemblySquare task.

    Pick up the nut, transport it above the square peg, place it on the
    peg. The peg location is absolute; the kinematic fake sim encodes
    the same absolute target so the success predicate agrees.
    """

    #: Absolute peg xy for the NutAssemblySquare task.
    PEG_XY: tuple[float, float] = (-0.12, -0.08)

    object_key = "object"

    def placement_target(self, state: np.ndarray) -> np.ndarray | None:  # noqa: ARG002
        config = self.config
        return np.array([self.PEG_XY[0], self.PEG_XY[1], config.placement_z])

    def is_success(self, state: np.ndarray) -> bool:
        """Mirror of env._check_success: nut on peg AND lifted."""
        obj = self.object_pos(state)
        if obj[2] <= self.config.success_z:
            return False
        placement = self._placement_target()
        assert placement is not None
        return bool(
            abs(obj[0] - placement[0]) < self.config.position_scale
            and abs(obj[1] - placement[1]) < self.config.position_scale
        )


class ScriptedToolHangController(ScriptedController):
    """Scripted oracle for the ToolHang task.

    Pick up the tool by its handle, transport it to the rack position,
    and hook it on the rack (release the gripper while aligned).
    """

    #: Absolute rack xy for the ToolHang task.
    RACK_XY: tuple[float, float] = (0.20, -0.05)

    object_key = "object"

    def placement_target(self, state: np.ndarray) -> np.ndarray | None:  # noqa: ARG002
        config = self.config
        return np.array([self.RACK_XY[0], self.RACK_XY[1], config.placement_z])

    def is_success(self, state: np.ndarray) -> bool:
        """Mirror of env._check_success: tool on rack AND lifted."""
        obj = self.object_pos(state)
        if obj[2] <= self.config.success_z:
            return False
        placement = self._placement_target()
        assert placement is not None
        return bool(
            abs(obj[0] - placement[0]) < self.config.position_scale
            and abs(obj[1] - placement[1]) < self.config.position_scale
        )


class ScriptedTransportController(ScriptedController):
    """Dimensional smoke controller for the TwoArmTransport task.

    Transport uses two Panda arms and the full state-only dataset has a
    59-dimensional observation and 14-dimensional action. This controller
    drives only robot0 and leaves robot1 open/neutral, so it is useful for
    action-shape smoke tests but is not a valid Gate-4 success oracle. A
    final five-task behavioral claim requires a coordinated two-arm
    controller before Transport can be admitted to the strict scripted gate.
    """

    #: Absolute bin xy for the single-arm Transport approximation.
    BIN_XY: tuple[float, float] = (-0.18, 0.12)

    object_key = "object"

    def placement_target(self, state: np.ndarray) -> np.ndarray | None:  # noqa: ARG002
        config = self.config
        return np.array([self.BIN_XY[0], self.BIN_XY[1], config.placement_z])

    def is_success(self, state: np.ndarray) -> bool:
        """Mirror of env._check_success: object in bin AND lifted."""
        obj = self.object_pos(state)
        if obj[2] <= self.config.success_z:
            return False
        placement = self._placement_target()
        assert placement is not None
        return bool(
            abs(obj[0] - placement[0]) < self.config.position_scale
            and abs(obj[1] - placement[1]) < self.config.position_scale
        )


# Backward-compat alias: the original name before the five-task refactor.
ScriptedLiftConfig = ScriptedControllerConfig


__all__ = [
    "ScriptedController",
    "ScriptedLiftController",
    "ScriptedCanController",
    "ScriptedSquareController",
    "ScriptedToolHangController",
    "ScriptedTransportController",
    "ScriptedControllerConfig",
    "ScriptedLiftConfig",
    "POSITION_SCALE",
    "ORIENTATION_SCALE",
    "SUCCESS_Z",
]
