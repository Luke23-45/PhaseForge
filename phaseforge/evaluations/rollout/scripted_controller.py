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

* ``grasp_pos(state)`` -- 3D point to approach and grasp (by default the
  manipulated object's state-vector position).
* ``placement_target(state)`` -- 3D end-effector point that positions the
  manipulated object at its receptacle. ``None`` means the task has no
  separate placement phase (e.g. Lift only needs to lift).
* ``is_success(state)`` -- environment's success predicate (mirrored from
  ``env._check_success()``).

The task-specific object layouts are defined by the canonical registry. Lift
stores the manipulated object's absolute position at ``object[0:3]``;
Can/Square store it at ``object[7:10]`` after the relative pose; ToolHang
stores the tool's absolute position at ``object[35:38]`` after the stand and
frame entries. The :class:`~phaseforge.evaluations.envs.task_registry.TaskSpec`
for each task carries the canonical keys/dims; this file is concerned only
with policy behavior and these explicitly pinned position offsets.

Phase progression for tasks with a placement phase (Can, Square, ToolHang):

    APPROACH -> DESCEND -> GRASP_HOLD -> LIFT -> TRANSPORT -> PLACE -> RETRACT

Phase progression for tasks without placement (Lift):

    APPROACH -> DESCEND -> GRASP_HOLD -> LIFT

A stall watchdog abandons an episode only when the distance to the active
target stops decreasing while tracking it. The GRASP phase is exempt
(closing the gripper intentionally keeps the end-effector stationary). The
abandoned episode then simply times out as a task failure, never as a
policy / simulator error.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from phaseforge.evaluations.envs.robosuite_adapter import StateSpec

#: Default Panda OSC output limits (matching the dataset's action contract).
POSITION_SCALE: float = 0.05
ORIENTATION_SCALE: float = 0.5

#: Lift geometry: table top at 0.8; success = cube z > 0.84.
TABLE_HEIGHT: float = 0.8
SUCCESS_Z: float = TABLE_HEIGHT + 0.04
OBJECT_HALF_SIZE: float = 0.0215

#: robosuite PandaGripper convention: -1 opens, +1 closes.
#: This is the normalized action passed through the pinned robosuite
#: controller, not a qpos sign heuristic.
GRIPPER_OPEN: float = -1.0
GRIPPER_CLOSE: float = 1.0

#: Steps to hold the gripper closed before lifting.
GRASP_HOLD_STEPS: int = 25

#: Steps to settle above the placement target while keeping the object held
#: before releasing it.
PLACE_HOLD_STEPS: int = 10

#: If the end-effector makes no progress within this many steps while
#: tracking a target, the episode is abandoned (stays in the final phase
#: with zero translation commands until the horizon expires -- recorded as
#: a task timeout, never as a simulator/policy failure).
STALL_STEPS: int = 30
# The watchdog compares target-distance progress rather than raw end-effector
# displacement.  OSC_POSE can converge in sub-millimetre steps near a target;
# raw displacement would incorrectly classify that valid convergence as a
# stall.  The fixed rollout horizon remains the authoritative task-time limit.
STALL_PROGRESS: float = 1e-4

#: Default placement target height (relative to the table) for tasks that
#: transport an object to a separate receptacle.
DEFAULT_PLACEMENT_Z: float = 0.95

# The Panda eef site is the gripper center. For the 4.3 cm cube, grasping is
# performed just above the cube center/top rather than at the high approach
# waypoint. This is also the geometry used by the kinematic gate simulator.
GRASP_Z_OFFSET: float = OBJECT_HALF_SIZE + 0.02
# Validated against all 50 cases in the pinned Lift reset bank. Other tasks
# retain GRASP_Z_OFFSET until their own scripted gates are validated.
LIFT_GRASP_Z_OFFSET: float = 0.01
# SquareNut is grasped around its handle / center plane. A positive offset
# places the Panda eef above the thin nut and misses native contact.
SQUARE_GRASP_Z_OFFSET: float = 0.0
# SquareNut contact is transient under the default OSC settling dynamics;
# validate it promptly before the thin nut slips from the closed gripper.
SQUARE_GRASP_HOLD_STEPS: int = 5
# A full OSC lift delta pulls the thin nut out of transient contact. Limit
# only Square's vertical lift command; approach, transport, and release keep
# the normal normalized action contract.
SQUARE_LIFT_ACTION_LIMIT: float = 0.2


class _Phase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    TRANSPORT = auto()
    PLACE = auto()
    RETRACT = auto()
    STALLED = auto()


@dataclass(frozen=True)
class ScriptedControllerConfig:
    """Tunables for the scripted controllers (defaults = validated values)."""

    position_scale: float = POSITION_SCALE
    orientation_scale: float = ORIENTATION_SCALE
    success_z: float = SUCCESS_Z
    descend_z_offset: float = GRASP_Z_OFFSET
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

    Subclasses override :meth:`object_pos`, :meth:`grasp_pos`,
    :meth:`placement_target`, and
    :meth:`is_success` to specialize for a specific task. Tasks that do
    not require a separate placement step (Lift) leave
    :meth:`placement_target` returning ``None``; the phase machine then
    stops at :attr:`_Phase.LIFT` and waits for the success predicate.
    """

    object_key: str = "object"
    """State-key carrying the manipulated object's pose (default: ``"object"``)."""

    #: Slice of the task's object vector containing the manipulated object's
    #: absolute xyz position. The default is the Lift layout.
    object_position_slice: tuple[int, int] = (0, 3)

    eef_key: str = "robot0_eef_pos"
    """State-key carrying the end-effector position (default: ``"robot0_eef_pos"``)."""

    def __init__(
        self,
        state_spec: StateSpec,
        *,
        config: ScriptedControllerConfig | None = None,
        env: Any | None = None,
    ) -> None:
        self.state_spec = state_spec
        # The oracle may use simulator geometry (target-bin / peg / hook
        # poses) but never pixels. Learned policies still receive only the
        # declared state vector. Keeping geometry on the oracle side avoids
        # hard-coded coordinates that silently break under a pinned reset
        # distribution.
        self.env = env
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
        self._last_target: np.ndarray | None = None
        self._last_target_distance: float | None = None
        self._stalled_from_phase: str | None = None
        self._placement_snapshot: np.ndarray | None = None

    @property
    def phase_name(self) -> str:
        """Stable diagnostic name for the current controller phase."""
        return self._phase.name

    @property
    def stalled_from_phase(self) -> str | None:
        """Phase in which the watchdog declared a stall, if any."""
        return self._stalled_from_phase

    # ------------------------------------------------------------------
    # State parsing
    # ------------------------------------------------------------------

    def eef_pos(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state[self.eef_start : self.eef_end], dtype=np.float64)

    def object_pos(self, state: np.ndarray) -> np.ndarray:
        """3D position of the manipulated object.

        Task subclasses set :attr:`object_position_slice` when the absolute
        position is not at the start of the ``object`` key.

        If a declared object key carries fewer than 3 dimensions, the
        controller falls back to the end-effector as a stand-in. This is only
        a defensive compatibility path; it is not a task-success oracle.
        """
        if self._object_is_indicator:
            return self.eef_pos(state).copy()
        obj = np.asarray(state[self.obj_start : self.obj_end], dtype=np.float64)
        start, end = self.object_position_slice
        if end <= obj.shape[0] and obj.shape[0] != 10:
            return obj[start:end].copy()
        # Compatibility path for the legacy kinematic unit fake, which uses
        # the Lift object layout for every task. Real task schemas are checked
        # during ingestion and therefore cannot silently take this path.
        return obj[:3].copy()

    def grasp_pos(self, state: np.ndarray) -> np.ndarray:
        """World-space point used for the approach and grasp descent.

        Most tasks grasp near the manipulated object's state-vector position.
        Tasks with an explicit simulator handle site can override this hook
        without changing the state-only policy interface.
        """
        return self.object_pos(state)

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
        grasp = self.grasp_pos(state)
        placement = self._placement_target()
        has_placement = placement is not None

        if self.is_success(state):
            return self._hold_action(GRIPPER_CLOSE if not has_placement else GRIPPER_OPEN)

        if self._phase is _Phase.STALLED:
            return self._hold_action(
                GRIPPER_CLOSE if not has_placement and self._approach_done else GRIPPER_OPEN
            )

        if self._phase is _Phase.APPROACH and not self._approach_done:
            target = grasp + np.array([0.0, 0.0, config.approach_z_offset])
            if np.linalg.norm(target - eef) < config.position_tolerance:
                self._approach_done = True
                self._phase = _Phase.DESCEND
            else:
                return self._track(target, GRIPPER_OPEN, eef, t)

        if self._phase is _Phase.DESCEND:
            target = grasp + np.array([0.0, 0.0, config.descend_z_offset])
            xy_dist = float(np.linalg.norm(target[:2] - eef[:2]))
            dist_3d = float(np.linalg.norm(target - eef))
            vertical_dist = abs(float(eef[2] - target[2]))
            if dist_3d < config.position_tolerance or (
                xy_dist < config.position_tolerance
                and vertical_dist < config.position_tolerance
            ):
                self._phase = _Phase.GRASP
                self._grasp_started_at = t
            else:
                return self._track(target, GRIPPER_OPEN, eef, t)

        if self._phase is _Phase.GRASP:
            assert self._grasp_started_at is not None
            if t - self._grasp_started_at >= config.grasp_hold_steps:
                # Do not let a fixed timer claim that a real grasp happened.
                # Unit-test fakes do not expose this predicate and return
                # None, while a real robosuite environment can veto the
                # transition when contact was not established.
                native_grasp = self._native_grasp_status()
                if native_grasp is False:
                    return self._hold_action(GRIPPER_CLOSE)
                # Snapshot the placement target at the moment the grasp
                # completes. The target must be relative to the object's
                # position *at the time of the grasp*, not the current
                # (already grasped, already moving) object position; the
                # latter would chase the eef indefinitely.
                self._snapshot_placement_target(state)
                self._phase = _Phase.LIFT
            else:
                return self._hold_action(GRIPPER_CLOSE)

        if self._phase is _Phase.LIFT:
            target = np.array([eef[0], eef[1], config.lift_z])
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
            return self._track(target, GRIPPER_CLOSE, eef, t)

        if self._phase is _Phase.TRANSPORT and has_placement and placement is not None:
            assert placement is not None
            # Move over the receptacle at the lifted height first. Descending
            # toward the final placement point while crossing the bin wall
            # can make the gripper collide with the receptacle and stall
            # before reaching its xy target. The separate PLACE phase then
            # performs the vertical descent and release.
            transport_target = np.array(
                [placement[0], placement[1], max(float(placement[2]), config.lift_z)],
                dtype=np.float64,
            )
            if (
                abs(placement[0] - eef[0]) < config.position_scale
                and abs(placement[1] - eef[1]) < config.position_scale
            ):
                self._phase = _Phase.PLACE
                self._place_started_at = t
            else:
                return self._track(transport_target, GRIPPER_CLOSE, eef, t)

        if self._phase is _Phase.PLACE and placement is not None:
            assert self._place_started_at is not None
            target_reached = (
                np.linalg.norm(placement - eef) <= config.position_tolerance
            )
            if (
                target_reached
                and t - self._place_started_at >= config.place_hold_steps
            ):
                self._phase = _Phase.RETRACT
                retract_target = np.array(
                    [
                        placement[0],
                        placement[1],
                        max(
                            float(placement[2]) + 0.10,
                            config.lift_z + config.approach_z_offset,
                        ),
                    ],
                    dtype=np.float64,
                )
                return self._track(retract_target, GRIPPER_OPEN, eef, t)
            return self._track(placement, GRIPPER_CLOSE, eef, t)

        if self._phase is _Phase.RETRACT and placement is not None:
            retract_target = np.array(
                [
                    placement[0],
                    placement[1],
                    max(
                        float(placement[2]) + 0.10,
                        config.lift_z + config.approach_z_offset,
                    ),
                ],
                dtype=np.float64,
            )
            return self._track(retract_target, GRIPPER_OPEN, eef, t)

        return self._hold_action(GRIPPER_OPEN)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _native_grasp_status(self) -> bool | None:
        """Return the simulator grasp predicate when this env exposes it.

        ``None`` means that the controller is running against a unit-test
        fake or an environment whose object geometry is not registered here.
        A definite ``False`` from a real environment is not treated as a
        successful grasp. This privileged signal is used only by the
        training-free rollout oracle, never by learned policies.
        """
        if self.env is None:
            return None
        checker = getattr(self.env, "_check_grasp", None)
        robots = getattr(self.env, "robots", None)
        gripper = getattr(robots[0], "gripper", None) if robots else None
        object_geoms = self._native_grasp_object_geoms()
        if not callable(checker) or gripper is None or object_geoms is None:
            return None
        try:
            return bool(checker(gripper=gripper, object_geoms=object_geoms))
        except Exception:  # noqa: BLE001 - a diagnostic guard must not crash execution
            return None

    def _native_grasp_object_geoms(self) -> Any | None:
        """Resolve the contact geoms for this task's manipulated object.

        robosuite exposes task objects through different attributes. Lift
        exposes ``env.cube``; PickPlaceCan stores the active object in the
        indexed ``env.objects`` list; other single-arm tasks commonly expose
        ``nut`` or ``tool``. The oracle may use these simulator-native
        geometries, but learned policies never receive them.
        """
        if self.env is None:
            return None

        candidates: list[Any] = []
        explicit_attr = getattr(self, "grasp_object_attr", None)
        if explicit_attr:
            explicit_object = getattr(self.env, explicit_attr, None)
            if explicit_object is not None:
                # Lift's robosuite task exposes ``cube`` in the form already
                # accepted by ``_check_grasp`` rather than as a model object.
                for geom_attr in ("contact_geoms", "handle_geoms"):
                    geoms = getattr(explicit_object, geom_attr, None)
                    if geoms is not None:
                        return geoms
                return explicit_object

        objects = getattr(self.env, "objects", None)
        object_id = getattr(self.env, "object_id", None)
        if objects is not None and object_id is not None:
            try:
                candidates.append(objects[int(object_id)])
            except (IndexError, KeyError, TypeError, ValueError):
                pass

        # NutAssembly stores the active Square / Round nut separately from
        # PickPlace's ``objects`` list. Resolve the selected nut so the
        # controller cannot advance on a timer while holding nothing.
        nuts = getattr(self.env, "nuts", None)
        nut_id = getattr(self.env, "nut_id", None)
        if nuts is not None and nut_id is not None:
            try:
                candidates.append(nuts[int(nut_id)])
            except (IndexError, KeyError, TypeError, ValueError):
                pass

        for attr in ("can", "nut", "tool", "cube"):
            candidates.append(getattr(self.env, attr, None))

        for obj in candidates:
            if obj is None:
                continue
            for geom_attr in ("contact_geoms", "handle_geoms"):
                geoms = getattr(obj, geom_attr, None)
                if geoms is not None:
                    return geoms
        return None

    def _track(self, target: np.ndarray, gripper: float, eef: np.ndarray, t: int) -> np.ndarray:
        # Run the watchdog only for an active tracking command. Phase
        # transition checks above get first chance to recognize that the
        # target has already been reached. PLACE is deliberately excluded:
        # it is a bounded contact-sensitive settling/release phase, and OSC
        # contact can temporarily stop reducing the eef-to-target distance.
        # PLACE remains bounded by the rollout horizon, but release is gated
        # on actually reaching the pinned end-effector target and then
        # holding there for place_hold_steps. The simulator's success
        # predicate remains the only success signal.
        if self._phase is not _Phase.PLACE:
            self._watchdog(eef, target, t)
        if self._phase is _Phase.STALLED:
            placement = self._placement_target()
            return self._hold_action(
                GRIPPER_CLOSE if placement is None and self._approach_done else GRIPPER_OPEN
            )
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

    def _env_site_pos(self, env: Any, site_id: int) -> np.ndarray:
        return np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).copy()

    def _env_body_pos(self, env: Any, body_id: int) -> np.ndarray:
        return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).copy()

    def _watchdog(self, eef: np.ndarray, target: np.ndarray, t: int) -> None:
        """Abandon an episode when tracking makes no progress.

        Progress is measured as a reduction in distance to ``target`` rather
        than as raw end-effector displacement. A phase can legitimately
        converge with very small Cartesian steps near its target.
        """
        config = self.config
        target = np.asarray(target, dtype=np.float64)
        distance = float(np.linalg.norm(target - eef))
        if distance <= config.position_scale:
            self._stall_since = None
            self._last_target = target.copy()
            self._last_target_distance = distance
            return

        target_changed = self._last_target is None or not np.allclose(
            target, self._last_target, rtol=0.0, atol=1e-9
        )
        if target_changed or self._last_target_distance is None:
            self._last_target = target.copy()
            self._last_target_distance = distance
            self._stall_since = t
            return

        progress = self._last_target_distance - distance
        self._last_target_distance = distance
        if progress < config.stall_progress:
            if self._stall_since is None:
                self._stall_since = t
            elif t - self._stall_since >= config.stall_steps:
                self._stalled_from_phase = self._phase.name
                self._phase = _Phase.STALLED
        else:
            self._stall_since = None


class ScriptedLiftController(ScriptedController):
    """Scripted oracle for the Lift task: pick up the cube and lift it.

    Success predicate: object z > 0.84 (above table by 4 cm).
    No placement phase -- the task is complete once lifted.
    """

    object_key = "object"
    grasp_object_attr = "cube"

    def __init__(
        self,
        state_spec: StateSpec,
        *,
        config: ScriptedControllerConfig | None = None,
        env: Any | None = None,
    ) -> None:
        if config is None:
            config = ScriptedControllerConfig(descend_z_offset=LIFT_GRASP_Z_OFFSET)
        super().__init__(state_spec, config=config, env=env)

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
    # Fallback only for the unit-test kinematic simulator. Real rollouts use
    # the pinned PickPlaceCan env's target-bin geometry below.
    RECEPTACLE_XY: tuple[float, float] = (0.15, 0.15)

    object_key = "object"
    object_position_slice = (7, 10)

    def placement_target(self, state: np.ndarray) -> np.ndarray | None:  # noqa: ARG002
        config = self.config
        if self.env is not None:
            placements = getattr(self.env, "target_bin_placements", None)
            object_id = getattr(self.env, "object_id", None)
            if placements is not None and object_id is not None:
                target = np.asarray(placements[int(object_id)], dtype=np.float64)
                return np.array([target[0], target[1], target[2] + 0.06])
            if hasattr(self.env, "bin2_pos"):
                target = np.asarray(self.env.bin2_pos, dtype=np.float64).copy()
                return np.array([target[0], target[1], config.placement_z])
        return np.array([self.RECEPTACLE_XY[0], self.RECEPTACLE_XY[1], config.placement_z])

    def is_success(self, state: np.ndarray) -> bool:
        """Do not substitute an xy proxy for the environment predicate.

        The adapter owns the real ``PickPlaceCan._check_success`` result;
        this controller therefore keeps tracking until the runner observes
        simulator success.
        """
        return False


class ScriptedSquareController(ScriptedController):
    """Scripted oracle for the NutAssemblySquare task.

    Pick up the nut, transport it above the square peg, place it on the
    peg. The peg location is absolute; the kinematic fake sim encodes
    the same absolute target so the success predicate agrees.
    """

    #: Absolute peg xy for the NutAssemblySquare task.
    PEG_XY: tuple[float, float] = (-0.12, -0.08)
    #: The square nut center must sit slightly above the peg body. The
    #: controller target is the eef center, so include the grasp offset below.
    PEG_OBJECT_Z_OFFSET: float = 0.01

    object_key = "object"
    object_position_slice = (7, 10)

    def __init__(
        self,
        state_spec: StateSpec,
        *,
        config: ScriptedControllerConfig | None = None,
        env: Any | None = None,
    ) -> None:
        if config is None:
            config = ScriptedControllerConfig(
                descend_z_offset=SQUARE_GRASP_Z_OFFSET,
                grasp_hold_steps=SQUARE_GRASP_HOLD_STEPS,
            )
        super().__init__(state_spec, config=config, env=env)

    def grasp_pos(self, state: np.ndarray) -> np.ndarray:
        """Use NutAssembly's active handle site for approach and descent.

        The SquareNut state position is its body/reference position, while
        robosuite's grasp geometry is the handle site. The site is resolved
        from the pinned environment at rollout time; the learned policy
        remains strictly state-only.
        """
        if self.env is not None:
            site_ids = getattr(self.env, "object_site_ids", None)
            nut_id = getattr(self.env, "nut_id", None)
            if site_ids is not None and nut_id is not None:
                try:
                    return self._env_site_pos(self.env, int(site_ids[int(nut_id)]))
                except (IndexError, KeyError, TypeError, ValueError):
                    pass
        return self.object_pos(state)

    def _normalized_action(
        self, delta: np.ndarray, gripper: float
    ) -> np.ndarray:
        action = super()._normalized_action(delta, gripper)
        if self._phase is _Phase.LIFT:
            action[2] = np.clip(
                action[2], -SQUARE_LIFT_ACTION_LIMIT, SQUARE_LIFT_ACTION_LIMIT
            )
        return action

    def placement_target(self, state: np.ndarray) -> np.ndarray | None:
        config = self.config
        if self.env is not None and hasattr(self.env, "peg1_body_id"):
            peg = self._env_body_pos(self.env, int(self.env.peg1_body_id))
            object_z = peg[2] + self.PEG_OBJECT_Z_OFFSET
            # placement_target is an end-effector target, while robosuite's
            # success predicate checks the nut body position. Preserve the
            # grasp-time eef-to-body offset so the body, rather than the eef,
            # is brought onto the peg.
            object_target = np.array([peg[0], peg[1], object_z])
            eef_to_object = self.eef_pos(state) - self.object_pos(state)
            return object_target + eef_to_object
        object_z = TABLE_HEIGHT + self.PEG_OBJECT_Z_OFFSET
        return np.array(
            [self.PEG_XY[0], self.PEG_XY[1], object_z + config.descend_z_offset]
        )

    def is_success(self, state: np.ndarray) -> bool:
        """Leave the complete peg predicate to the simulator adapter."""
        return False


class ScriptedToolHangController(ScriptedController):
    """Scripted oracle for the ToolHang task.

    Pick up the tool by its handle, transport it to the rack position,
    and hook it on the rack (release the gripper while aligned).
    """

    #: Absolute rack xy for the ToolHang task.
    RACK_XY: tuple[float, float] = (0.20, -0.05)

    object_key = "object"
    # ToolHang stores stand, frame, and tool entries as three consecutive
    # 14-D blocks. Each block is relative pose (7) followed by absolute pose
    # (7), so the tool absolute xyz is object[28 + 7 : 28 + 10].
    object_position_slice = (35, 38)

    def placement_target(self, state: np.ndarray) -> np.ndarray | None:  # noqa: ARG002
        config = self.config
        if self.env is not None and hasattr(self.env, "obj_site_id"):
            site_id = int(self.env.obj_site_id["frame_hang_site"])
            hook = self._env_site_pos(self.env, site_id)
            body_ids = getattr(self.env, "obj_body_id", {})
            tool_body_id = body_ids.get("tool")
            if tool_body_id is not None:
                tool_pos = self._env_body_pos(self.env, int(tool_body_id))
                hole_pos = self._env_site_pos(
                    self.env, int(self.env.obj_site_id["tool_hole1_center"])
                )
                # Preserve the current tool orientation while transporting;
                # this converts the hook target from hole position to tool
                # body position without assuming a hard-coded tool offset.
                return hook - (hole_pos - tool_pos)
            # Move the tool hole to the hook line while keeping a small
            # clearance above it; the env's _check_success remains the only
            # success predicate. This is a geometry-aware oracle, not a
            # simplified xy/z success proxy.
            return np.array([hook[0], hook[1], hook[2] + 0.02])
        return np.array([self.RACK_XY[0], self.RACK_XY[1], config.placement_z])

    def is_success(self, state: np.ndarray) -> bool:
        """Leave the contact-and-geometry predicate to the simulator."""
        return False


class ScriptedTransportController(ScriptedController):
    """Coordinated state oracle for the TwoArmTransport task.

    Transport uses two Panda arms and the full state-only dataset has a
    59-dimensional observation and 14-dimensional action. This controller
    The 41-dimensional transport object state exposes payload/trash and both
    target-bin positions. The controller moves the payload with robot0 and
    removes trash with robot1, then releases both objects in their respective
    bins. It uses no images and the adapter still decides success via the
    environment's own ``_check_success`` predicate.
    """

    #: Absolute bin xy for the single-arm Transport approximation.
    BIN_XY: tuple[float, float] = (-0.18, 0.12)

    def __init__(self, state_spec: StateSpec, **kwargs) -> None:
        super().__init__(state_spec, **kwargs)
        if not self._two_arm:
            return  # retained for the legacy single-arm kinematic unit fake
        if self.obj_end - self.obj_start != 41:
            raise ValueError(
                "Transport oracle requires the registered 41-dimensional "
                "object state; refusing a single-object approximation."
            )
        self.eef1_start, self.eef1_end = state_spec.index_of("robot1_eef_pos")

    def reset(self) -> None:
        super().reset()
        self._transport_phase = _Phase.APPROACH
        self._transport_started_at: int | None = None

    def _transport_values(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obj = np.asarray(state[self.obj_start : self.obj_end], dtype=np.float64)
        return obj[0:3], obj[7:10], obj[21:24], obj[24:27]

    def _transport_action(
        self,
        state: np.ndarray,
        t: int,
    ) -> np.ndarray:
        payload, trash, target_bin, trash_bin = self._transport_values(state)
        eef0 = self.eef_pos(state)
        eef1 = np.asarray(state[self.eef1_start : self.eef1_end], dtype=np.float64)
        payload_done = bool(state[self.obj_start + 27] > 0.5)
        trash_done = bool(state[self.obj_start + 28] > 0.5)
        if payload_done and trash_done:
            return self._hold_action(GRIPPER_OPEN)

        # The first two phases align both grippers with their objects. The
        # objects are absolute positions in the published object vector.
        if self._transport_phase is _Phase.APPROACH:
            targets = (
                payload + np.array([0.0, 0.0, self.config.approach_z_offset]),
                trash + np.array([0.0, 0.0, self.config.approach_z_offset]),
            )
            if (
                max(
                    np.linalg.norm(targets[0] - eef0),
                    np.linalg.norm(targets[1] - eef1),
                )
                < self.config.position_tolerance
            ):
                self._transport_phase = _Phase.DESCEND
            return self._two_arm_action(targets, eef0, eef1, GRIPPER_OPEN)

        if self._transport_phase is _Phase.DESCEND:
            targets = (
                payload + np.array([0.0, 0.0, self.config.descend_z_offset]),
                trash + np.array([0.0, 0.0, self.config.descend_z_offset]),
            )
            if (
                max(
                    np.linalg.norm(targets[0] - eef0),
                    np.linalg.norm(targets[1] - eef1),
                )
                < self.config.position_tolerance
            ):
                self._transport_phase = _Phase.GRASP
                self._transport_started_at = t
            return self._two_arm_action(targets, eef0, eef1, GRIPPER_OPEN)

        if self._transport_phase is _Phase.GRASP:
            assert self._transport_started_at is not None
            if t - self._transport_started_at >= self.config.grasp_hold_steps:
                self._transport_phase = _Phase.LIFT
            return self._two_arm_action((eef0, eef1), eef0, eef1, GRIPPER_CLOSE)

        if self._transport_phase is _Phase.LIFT:
            z = max(float(target_bin[2]), float(trash_bin[2])) + self.config.lift_z - TABLE_HEIGHT
            targets = (np.array([payload[0], payload[1], z]), np.array([trash[0], trash[1], z]))
            if min(eef0[2], eef1[2]) >= z - self.config.position_scale:
                self._transport_phase = _Phase.TRANSPORT
            return self._two_arm_action(targets, eef0, eef1, GRIPPER_CLOSE)

        targets = (
            target_bin + np.array([0.0, 0.0, self.config.descend_z_offset + 0.02]),
            trash_bin + np.array([0.0, 0.0, self.config.descend_z_offset + 0.02]),
        )
        if self._transport_phase is _Phase.TRANSPORT:
            if (
                max(
                    np.linalg.norm(targets[0] - eef0),
                    np.linalg.norm(targets[1] - eef1),
                )
                < self.config.position_scale
            ):
                self._transport_phase = _Phase.PLACE
                self._transport_started_at = t
            return self._two_arm_action(targets, eef0, eef1, GRIPPER_CLOSE)
        assert self._transport_phase is _Phase.PLACE
        assert self._transport_started_at is not None
        if t - self._transport_started_at >= self.config.place_hold_steps:
            return self._hold_action(GRIPPER_OPEN)
        return self._two_arm_action(targets, eef0, eef1, GRIPPER_OPEN)

    def _two_arm_action(
        self,
        targets: tuple[np.ndarray, np.ndarray],
        eef0: np.ndarray,
        eef1: np.ndarray,
        gripper: float,
    ) -> np.ndarray:
        action = np.zeros(14, dtype=np.float64)
        action[0:3] = np.clip((targets[0] - eef0) / self.config.position_scale, -1.0, 1.0)
        action[6] = gripper
        action[7:10] = np.clip((targets[1] - eef1) / self.config.position_scale, -1.0, 1.0)
        action[13] = gripper
        return action

    def act(self, state: np.ndarray, t: int) -> np.ndarray:
        if self._two_arm:
            return self._transport_action(state, t)
        return super().act(state, t)

    object_key = "object"

    def placement_target(self, state: np.ndarray) -> np.ndarray | None:  # noqa: ARG002
        config = self.config
        return np.array([self.BIN_XY[0], self.BIN_XY[1], config.placement_z])

    def is_success(self, state: np.ndarray) -> bool:
        """Leave the two-object success predicate to the simulator."""
        return False


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
    "LIFT_GRASP_Z_OFFSET",
]
