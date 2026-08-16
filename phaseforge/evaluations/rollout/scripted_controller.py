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
# SquareNut contact is transient under the OSC settling dynamics. Hold long
# enough to confirm contact, then use a small vertical increment so the thin
# nut is not pulled out of the gripper on the first lift step.
SQUARE_GRASP_HOLD_STEPS: int = 5
# 0.05 corresponds to a nominal 2.5 mm OSC delta with the pinned position
# scale. Only Square's vertical lift command is limited; approach, transport,
# and release keep the normal normalized action contract.
SQUARE_LIFT_ACTION_LIMIT: float = 0.05
# Give the thin nut a few simulator steps to settle under a closed gripper
# before introducing any upward motion. Contact is rechecked on every step.
SQUARE_LIFT_SETTLE_STEPS: int = 10
# A single false contact result is not enough evidence that the thin nut was
# dropped.  OSC contact can flicker while the gripper is settling; hold still
# with the fingers closed for a short confirmation window before recovering.
SQUARE_GRASP_LOSS_CONFIRM_STEPS: int = 5
# Placement contact is more sensitive because the nut is colliding with the
# peg. Allow one false native-contact sample, but do not freeze the placement
# controller while waiting for confirmation.
SQUARE_PLACE_GRASP_LOSS_CONFIRM_STEPS: int = 2
# NutAssembly's reach term is ``1 - tanh(10 * distance) < 0.6``.  Keep the
# same threshold when deciding whether the released nut is clear of the eef.
SQUARE_SUCCESS_REACH_DISTANCE: float = float(np.arctanh(0.4) / 10.0)
# SquareNut's handle is fixed along the object's local +x axis.  The Panda
# gripper is symmetric under a pi yaw flip, so align to the nearest equivalent
# object yaw rather than commanding an unnecessarily long rotation.
SQUARE_YAW_ALIGNMENT_TOLERANCE: float = 0.03


class _Phase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    TRANSPORT = auto()
    PLACE = auto()
    RETRACT = auto()
    STALLED = auto()


class _TransportPhase(Enum):
    """Privileged oracle phases for the two-arm Transport task."""

    LID_APPROACH = auto()
    LID_DESCEND = auto()
    LID_GRASP = auto()
    LID_LIFT = auto()
    LID_CLEAR = auto()
    PAYLOAD_APPROACH = auto()
    PAYLOAD_DESCEND = auto()
    PAYLOAD_GRASP = auto()
    PAYLOAD_LIFT = auto()
    # --- MISSING HANDOVER PHASES ---
    TRASH_TRANSPORT = auto()
    TRASH_PLACE = auto()
    HANDOVER_APPROACH = auto()
    HANDOVER_DESCEND = auto()
    HANDOVER_GRASP = auto()
    HANDOVER_RELEASE = auto()
    PAYLOAD_TRANSPORT = auto()
    PAYLOAD_PLACE = auto()
    DONE = auto()
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
                and self._placement_release_ready(state)
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

    def _placement_release_ready(self, state: np.ndarray) -> bool:  # noqa: ARG002
        """Return whether a placement task may open its gripper.

        The base controller only has an end-effector target. Tasks whose
        simulator predicate evaluates the object body can override this hook
        to require the body itself to be in the receptacle before release.
        """
        return True

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
        self.eef_quat_start, self.eef_quat_end = state_spec.index_of(
            "robot0_eef_quat"
        )

    def reset(self) -> None:
        super().reset()
        self._square_lift_started_at: int | None = None
        self._square_grasp_loss_streak = 0
        self._square_place_grasp_loss_streak = 0
        self._square_yaw_error = 0.0

    @staticmethod
    def _yaw_from_xyzw(quat: np.ndarray) -> float:
        """Return global yaw for robosuite's ``xyzw`` quaternion layout."""
        quat = np.asarray(quat, dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-12:
            return 0.0
        x, y, z, w = quat / norm
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _update_yaw_error(self, state: np.ndarray) -> None:
        """Compute the shortest symmetric yaw correction for the gripper."""
        eef_quat = np.asarray(state[self.eef_quat_start : self.eef_quat_end])
        obj = np.asarray(state[self.obj_start : self.obj_end])
        # robomimic's Square object observation is relative pose (7) followed
        # by absolute pose (7); the absolute quaternion is obj[10:14].
        if eef_quat.shape != (4,) or obj.shape[0] < 14:
            self._square_yaw_error = 0.0
            return
        eef_yaw = self._yaw_from_xyzw(eef_quat)
        object_yaw = self._yaw_from_xyzw(obj[10:14])
        error = object_yaw - eef_yaw
        # The fingers are unchanged by a pi flip. Wrap into [-pi/2, pi/2).
        self._square_yaw_error = float((error + np.pi / 2.0) % np.pi - np.pi / 2.0)

    def _hold_action(self, gripper: float) -> np.ndarray:
        """Close / open while preserving the required handle yaw."""
        action = super()._hold_action(gripper)
        if (
            self._phase
            in (_Phase.APPROACH, _Phase.DESCEND, _Phase.GRASP, _Phase.LIFT, _Phase.TRANSPORT)
            and abs(self._square_yaw_error) > SQUARE_YAW_ALIGNMENT_TOLERANCE
        ):
            action[5] = np.clip(
                self._square_yaw_error / self.config.orientation_scale,
                -1.0,
                1.0,
            )
        return action

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
        if (
            self._phase
            in (_Phase.APPROACH, _Phase.DESCEND, _Phase.GRASP, _Phase.LIFT, _Phase.TRANSPORT)
            and abs(self._square_yaw_error) > SQUARE_YAW_ALIGNMENT_TOLERANCE
        ):
            action[5] = np.clip(
                self._square_yaw_error / self.config.orientation_scale,
                -1.0,
                1.0,
            )
        if self._phase is _Phase.LIFT:
            action[2] = np.clip(
                action[2], -SQUARE_LIFT_ACTION_LIMIT, SQUARE_LIFT_ACTION_LIMIT
            )
        return action

    def act(self, state: np.ndarray, t: int) -> np.ndarray:
        """Track the nut body to the peg while the nut is held.

        The nut is grasped at its handle, while robosuite's placement
        predicate evaluates the nut root-body position.  A single
        grasp-time ``eef - body`` offset is not invariant under the small
        rotations introduced by OSC contact dynamics.  Once the controller
        has verified the grasp, refresh the end-effector target from the
        current low-dimensional body state before each held LIFT / TRANSPORT
        / PLACE step.  This is closed-loop state feedback, not a success
        oracle: the transition to release is still governed by the base
        controller's target and the environment's own success predicate.
        """
        self._update_yaw_error(state)
        # During LIFT / TRANSPORT, debounce a false native contact result.
        # NutAssembly's OSC contact can flicker while the nut is settling; an
        # immediate reset moves the eef away and turns a recoverable contact
        # blip into a real drop.  If the false result persists, recover by
        # reacquiring the current handle instead of following a stale target.
        if self._phase in (_Phase.LIFT, _Phase.TRANSPORT):
            self._square_place_grasp_loss_streak = 0
            native_grasp = self._native_grasp_status()
            if native_grasp is False:
                self._square_grasp_loss_streak += 1
                if self._square_grasp_loss_streak < SQUARE_GRASP_LOSS_CONFIRM_STEPS:
                    return self._hold_action(GRIPPER_CLOSE)
                self._reset_square_grasp_recovery()
            else:
                self._square_grasp_loss_streak = 0
                self._placement_snapshot = self.placement_target(state)
        else:
            self._square_grasp_loss_streak = 0

        # Once in PLACE, a false contact is not an intentional release: the
        # base controller has not opened the gripper yet.  If the nut is not
        # already on the peg, reacquire it instead of holding closed forever
        # at an unreachable placement target.  If the authoritative geometry
        # says it is already placed, let the base controller open and retract.
        if self._phase is _Phase.PLACE:
            self._square_grasp_loss_streak = 0
            native_grasp = self._native_grasp_status()
            if native_grasp is False and not self._placement_release_ready(state):
                self._square_place_grasp_loss_streak += 1
                if (
                    self._square_place_grasp_loss_streak
                    >= SQUARE_PLACE_GRASP_LOSS_CONFIRM_STEPS
                ):
                    # Contact can flicker exactly as the nut reaches the peg,
                    # but persistent loss must recover promptly so the
                    # remaining horizon is not consumed in PLACE. The first
                    # false sample intentionally falls through to the normal
                    # closed placement action.
                    self._reset_square_grasp_recovery()
            else:
                self._square_place_grasp_loss_streak = 0
        else:
            self._square_place_grasp_loss_streak = 0

        phase_before = self._phase
        action = super().act(state, t)
        if phase_before is _Phase.GRASP and self._phase is _Phase.LIFT:
            self._square_lift_started_at = t
        if (
            self._phase is _Phase.LIFT
            and self._square_lift_started_at is not None
            and t - self._square_lift_started_at < SQUARE_LIFT_SETTLE_STEPS
        ):
            return self._hold_action(GRIPPER_CLOSE)
        return action

    def _reset_square_grasp_recovery(self) -> None:
        """Return to handle acquisition after a confirmed Square drop."""
        self._phase = _Phase.APPROACH
        self._approach_done = False
        self._grasp_started_at = None
        self._placement_snapshot = None
        self._stall_since = None
        self._last_target = None
        self._last_target_distance = None
        self._stalled_from_phase = None
        self._square_lift_started_at = None
        self._square_grasp_loss_streak = 0
        self._square_place_grasp_loss_streak = 0

    def _placement_release_ready(self, state: np.ndarray) -> bool:
        """Require the nut body to satisfy robosuite's placement geometry.

        ``NutAssembly._check_success`` evaluates the root body, not the
        handle or end-effector: the body must be close to the peg in xy and
        below the table-height success limit.  The additional eef/body
        separation preserves robosuite's reach term, which requires the
        gripper to be clear of the nut after release.
        """
        if self.env is None or not hasattr(self.env, "peg1_body_id"):
            return True
        peg = self._env_body_pos(self.env, int(self.env.peg1_body_id))
        obj = self.object_pos(state)
        eef = self.eef_pos(state)
        nut_id = int(getattr(self.env, "nut_id", 0))

        # Use robosuite's own NutAssembly geometry when available.  Its
        # on_peg predicate is the publication-facing task success contract;
        # duplicating it with a peg-relative z threshold was stricter and
        # caused the controller to wait forever after the simulator accepted
        # a valid placement.
        on_peg = getattr(self.env, "on_peg", None)
        if callable(on_peg):
            on_peg_result = bool(on_peg(obj, nut_id))
        else:
            table_offset = np.asarray(
                getattr(self.env, "table_offset", [0.0, 0.0, float(peg[2]) - 0.03]),
                dtype=np.float64,
            )
            on_peg_result = bool(
                np.all(np.abs(obj[:2] - peg[:2]) < 0.03)
                and obj[2] < table_offset[2] + 0.05
            )
        clear_of_gripper = (
            float(np.linalg.norm(eef - obj)) > SQUARE_SUCCESS_REACH_DISTANCE
        )
        return on_peg_result and clear_of_gripper

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

    def placement_target(self, state: np.ndarray) -> np.ndarray | None:
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
                tool_target = hook - (hole_pos - tool_pos)
                # The base phase machine tracks the end-effector, whereas
                # ToolHang's success predicate evaluates the tool hole. Keep
                # the grasp-time eef-to-tool-body offset so the tool body is
                # transported to the geometry-derived target.
                eef_to_tool = self.eef_pos(state) - self.object_pos(state)
                return tool_target + eef_to_tool
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
    """State-oracle controller for robosuite's two-arm Transport task.

    The real task is ordered: remove the start-bin lid, pick the payload,
    remove the trash, and release both objects in their target bins. The
    controller uses only the declared low-dimensional state for target
    positions; simulator geometry is used only for the privileged grasp
    guard in this training-free gate oracle.
    """

    BIN_XY: tuple[float, float] = (-0.18, 0.12)
    LID_CLEAR_OFFSET = np.array([-0.22, 0.0, 0.05], dtype=np.float64)
    BIN_OBJECT_Z_OFFSET: float = 0.02

    #: Hover waypoint for arm1 above the trash before descending into the
    #: target bin. The trash sits inside the target bin (robosuite places it
    #: at ``target_bin_pos + wall_thickness``); a straight-line approach from
    #: the arm's rest pose to ``trash + approach_z_offset`` clips the bin
    #: wall because that point is below the wall top.  The waypoint is high
    #: enough above the bin walls for an unobstructed horizontal approach
    #: and capped at the same lift ceiling (1.08) the lid lift already uses,
    #: to stay within arm1's validated workspace.
    TRASH_APPROACH_Z_OFFSET: float = 0.24
    TRASH_APPROACH_Z_CAP: float = 1.08

    #: Trash grasp offset.  The 4 cm trash cube (~2 cm half-size) sits on the
    #: bin floor; placing the gripper site ~1 cm above the cube center mirrors
    #: :data:`LIFT_GRASP_Z_OFFSET` so the fingers close around the upper part
    #: of the cube rather than hovering above it (the previous Lift-sized
    #: ``descend_z_offset`` of 0.0415 left the cube just below the finger pads
    #: and ``check_grasp`` never registered contact).
    TRASH_GRASP_Z_OFFSET: float = 0.01

    #: Extra steps the LID_GRASP/PAYLOAD_GRASP hold is given before declaring
    #: a definite native-grasp failure. The native checks run every step
    #: after ``grasp_hold_steps``; this window bounds the wasted horizon when
    #: the geometry fix does not establish contact on a specific reset.
    GRASP_CONFIRM_STEPS: int = 60

    #: Local-frame z offset of the hammer head center from the composite
    #: body center: ``handle_length/2 + head_halfsize`` for the TwoArmTransport
    #: HammerObject (``handle_length=0.20``, ``handle_radius=0.015`` so
    #: ``head_halfsize in [0.015, 0.018]``; we use ``handle_radius`` as a
    #: conservative approximation).  This puts the grasp target on the head
    #: (a chunky 3.6x1.8x1.8 cm box) instead of the thin 3 cm handle, which
    #: makes the grasp robust to OSC xy drift of a few cm.
    PAYLOAD_HEAD_OFFSET_Z: float = 0.115

    #: Z offset above the hammer head center for the PAYLOAD_DESCEND target.
    #: The eef settles ABOVE the head so the OSC doesn't push the hammer
    #: sideways during the descent; the pads then close on the head from
    #: above during PAYLOAD_GRASP.
    PAYLOAD_HEAD_DESCEND_Z_OFFSET: float = 0.01

    #: Safety bound for PAYLOAD_DESCEND.  With the saturated OSC action the
    #: eef covers the ~10-15 cm descent in ~100-150 steps at the env's OSC
    #: speed; 150 is a generous upper bound that still fires before the
    #: 700-step horizon is consumed.  Without this bound, an unreachable
    #: target (e.g., the eef xy drifts too far off the head) would leave
    #: the controller stuck in DESCEND until the horizon expires.
    PAYLOAD_DESCEND_HOLD_STEPS: int = 150

    object_key = "object"

    def __init__(self, state_spec: StateSpec, **kwargs) -> None:
        super().__init__(state_spec, **kwargs)
        if not self._two_arm:
            return
        if self.obj_end - self.obj_start != 41:
            raise ValueError(
                "Transport oracle requires the registered 41-dimensional "
                "object state; refusing a single-object approximation."
            )
        self.eef1_start, self.eef1_end = state_spec.index_of("robot1_eef_pos")

    @staticmethod
    def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
        """Convert a robosuite (x, y, z, w) quaternion to a 3x3 rotation matrix.

        Mirrors the convention used by ``robosuite.utils.transform_utils.mat2quat``
        (which the transport env uses to expose ``payload_quat`` in the state).
        """
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _head_center(self, payload: np.ndarray, payload_quat: np.ndarray) -> np.ndarray:
        """World position of the hammer head center.

        ``payload_pos`` is the hammer's composite body center (mid-handle).
        The head sits at local ``(0, 0, handle_length/2 + head_halfsize)``
        on the hammer's local +z axis (the handle axis).  We rotate that
        local offset by the body orientation to get the head center in the
        world frame.
        """
        rot = self._quat_xyzw_to_mat(payload_quat)
        offset = rot @ np.array([0.0, 0.0, self.PAYLOAD_HEAD_OFFSET_Z], dtype=np.float64)
        return payload + offset

    def reset(self) -> None:
        super().reset()
        self._transport_phase = _TransportPhase.LID_APPROACH
        self._transport_started_at: int | None = None
        self._lid_clear_target: np.ndarray | None = None
        self._payload_eef_offset: np.ndarray | None = None
        self._trash_eef_offset: np.ndarray | None = None
        self._place_targets: tuple[np.ndarray, np.ndarray] | None = None
        self._lid_drop_started_at: int | None = None
        self._handover_head_target: np.ndarray | None = None

    @property
    def phase_name(self) -> str:
        """Expose the two-arm phase in gate reports and diagnostics."""
        if self._two_arm:
            return self._transport_phase.name
        return super().phase_name

    def _transport_values(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obj = np.asarray(state[self.obj_start : self.obj_end], dtype=np.float64)
        return obj[0:3], obj[7:10], obj[14:17], obj[21:24], obj[24:27]

    def _eef1_pos(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state[self.eef1_start : self.eef1_end], dtype=np.float64)

    def _both_reached(
        self,
        targets: tuple[np.ndarray, np.ndarray],
        eef0: np.ndarray,
        eef1: np.ndarray,
        tolerance: float | None = None,
    ) -> bool:
        tol = self.config.position_tolerance if tolerance is None else tolerance
        return max(
            np.linalg.norm(targets[0] - eef0),
            np.linalg.norm(targets[1] - eef1),
        ) <= tol

    def _native_transport_grasp(self, arm_index: int, object_name: str) -> bool | None:
        """Use robosuite's native contact predicate for a real grasp."""
        if self.env is None:
            return None
        checker = getattr(self.env, "_check_grasp", None)
        robots = getattr(self.env, "robots", None)
        objects = getattr(getattr(self.env, "transport", None), "objects", None)
        if not callable(checker) or robots is None or objects is None:
            return None
        if arm_index >= len(robots) or object_name not in objects:
            return None
        obj = objects[object_name]
        geoms = getattr(obj, "contact_geoms", None)
        gripper = getattr(robots[arm_index], "gripper", None)
        if geoms is None or gripper is None:
            return None
        try:
            return bool(checker(gripper=gripper, object_geoms=geoms))
        except Exception:  # noqa: BLE001 - diagnostic oracle must not crash
            return None

    def _transport_pad_contact(self, arm_index: int, object_name: str) -> bool | None:
        """True if any gripper fingerpad of ``arm_index`` contacts the object.

        Unlike :meth:`_native_transport_grasp` (which needs both pads in a
        closed pinch), this only needs a single pad contact.  Used to
        confirm the eef has reached the hammer during PAYLOAD_DESCEND,
        regardless of which orientation the hammer settled in.
        """
        if self.env is None:
            return None
        checker = getattr(self.env, "check_contact", None)
        robots = getattr(self.env, "robots", None)
        objects = getattr(getattr(self.env, "transport", None), "objects", None)
        if not callable(checker) or robots is None or objects is None:
            return None
        if arm_index >= len(robots) or object_name not in objects:
            return None
        # ``robot.gripper`` is a dict {arm_name: GripperModel} in the
        # opposed two-arm config; ``_native_transport_grasp`` happens to
        # work because ``env._check_grasp`` accepts the dict directly, but
        # ``check_contact`` needs a single GripperModel with ``important_geoms``.
        gripper_mapping = getattr(robots[arm_index], "gripper", None)
        if not isinstance(gripper_mapping, dict):
            return None
        arms = getattr(robots[arm_index], "arms", ())
        gripper = gripper_mapping.get(arms[0]) if arms else None
        if gripper is None:
            gripper = next(iter(gripper_mapping.values()), None)
        important = getattr(gripper, "important_geoms", None)
        if (
            important is None
            or "left_fingerpad" not in important
            or "right_fingerpad" not in important
        ):
            return None
        geoms = getattr(objects[object_name], "contact_geoms", None)
        if geoms is None:
            return None
        try:
            return any(
                bool(checker(pad, geoms))
                for pad in (important["left_fingerpad"], important["right_fingerpad"])
            )
        except Exception:  # noqa: BLE001 - diagnostic oracle must not crash
            return None

    def _two_arm_action(
        self,
        targets: tuple[np.ndarray, np.ndarray],
        eef0: np.ndarray,
        eef1: np.ndarray,
        grippers: tuple[float, float],
    ) -> np.ndarray:
        action = np.zeros(14, dtype=np.float64)
        action[0:3] = np.clip(
            (targets[0] - eef0) / self.config.position_scale, -1.0, 1.0
        )
        action[6] = grippers[0]
        action[7:10] = np.clip(
            (targets[1] - eef1) / self.config.position_scale, -1.0, 1.0
        )
        action[13] = grippers[1]
        return action

    def _transport_action(self, state: np.ndarray, t: int) -> np.ndarray:
        if self._transport_phase is _TransportPhase.STALLED:
            return self._two_arm_action(
                (self.eef_pos(state), self._eef1_pos(state)),
                self.eef_pos(state),
                self._eef1_pos(state),
                (GRIPPER_OPEN, GRIPPER_OPEN),
            )

        payload, trash, lid_handle, target_bin, trash_bin = self._transport_values(state)
        payload_quat = np.asarray(
            state[self.obj_start + 3 : self.obj_start + 7], dtype=np.float64
        )
        eef0 = self.eef_pos(state)
        eef1 = self._eef1_pos(state)
        hold1 = eef1.copy()

        if self._lid_clear_target is None:
            self._lid_clear_target = lid_handle + self.LID_CLEAR_OFFSET

        lid_approach = lid_handle + np.array([0.0, 0.0, self.config.approach_z_offset])
        lid_grasp = lid_handle.copy()
        trash_high = np.array(
            [
                trash[0],
                trash[1],
                min(trash[2] + self.TRASH_APPROACH_Z_OFFSET, self.TRASH_APPROACH_Z_CAP),
            ],
            dtype=np.float64,
        )
        trash_grasp = trash + np.array([0.0, 0.0, self.TRASH_GRASP_Z_OFFSET])

        if self._transport_phase is _TransportPhase.LID_APPROACH:
            targets = (lid_approach, trash_high)
            self._transport_watchdog(targets, eef0, eef1, t)
            if self._both_reached(targets, eef0, eef1):
                self._transport_phase = _TransportPhase.LID_DESCEND
                self._stall_since = None
                self._last_target = None
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_OPEN, GRIPPER_OPEN))

        if self._transport_phase is _TransportPhase.LID_DESCEND:
            targets = (lid_grasp, trash_grasp)
            self._transport_watchdog(targets, eef0, eef1, t)
            if self._both_reached(targets, eef0, eef1):
                self._transport_phase = _TransportPhase.LID_GRASP
                self._transport_started_at = t
                self._stall_since = None
                self._last_target = None
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_OPEN, GRIPPER_OPEN))

        if self._transport_phase is _TransportPhase.LID_GRASP:
            assert self._transport_started_at is not None
            held_for = t - self._transport_started_at
            if held_for >= self.config.grasp_hold_steps:
                lid_grasped = self._native_transport_grasp(0, "lid")
                trash_grasped = self._native_transport_grasp(1, "trash")
                if lid_grasped is not False and trash_grasped is not False:
                    self._transport_phase = _TransportPhase.LID_LIFT
                    self._stall_since = None
                    self._last_target = None
                elif (
                    lid_grasped is False or trash_grasped is False
                ) and held_for >= self.config.grasp_hold_steps + self.GRASP_CONFIRM_STEPS:
                    self._stalled_from_phase = self._transport_phase.name
                    self._transport_phase = _TransportPhase.STALLED
            return self._two_arm_action(
                (eef0, eef1), eef0, eef1, (GRIPPER_CLOSE, GRIPPER_CLOSE)
            )

        if self._transport_phase is _TransportPhase.LID_LIFT:
            lid_lift = np.array([eef0[0], eef0[1], max(eef0[2], 1.08)])
            targets = (lid_lift, hold1)
            self._transport_watchdog(targets, eef0, eef1, t)
            if eef0[2] >= 1.08 - self.config.position_scale:
                self._transport_phase = _TransportPhase.LID_CLEAR
                self._stall_since = None
                self._last_target = None
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_CLOSE, GRIPPER_CLOSE))

        if self._transport_phase is _TransportPhase.LID_CLEAR:
            targets = (self._lid_clear_target, hold1)
            self._transport_watchdog(targets, eef0, eef1, t)
            if np.linalg.norm(targets[0] - eef0) <= self.config.position_tolerance:
                if getattr(self, "_lid_drop_started_at", None) is None:
                    self._lid_drop_started_at = t
                if t - self._lid_drop_started_at >= 15:
                    self._transport_phase = _TransportPhase.PAYLOAD_APPROACH
                    self._stall_since = None
                    self._last_target = None
                return self._two_arm_action(
                    targets, eef0, eef1, (GRIPPER_OPEN, GRIPPER_CLOSE)
                )
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_CLOSE, GRIPPER_CLOSE))

        # Safe clearance above the wall
        payload_approach_z = max(float(payload[2]) + self.config.approach_z_offset, 1.08)
        payload_approach = np.array([payload[0], payload[1], payload_approach_z], dtype=np.float64)
        payload_descend = payload + np.array([0.0, 0.0, 0.01], dtype=np.float64)

        if self._transport_phase is _TransportPhase.PAYLOAD_APPROACH:
            targets = (payload_approach, hold1)
            self._transport_watchdog(targets, eef0, eef1, t)
            if self._both_reached(targets, eef0, eef1):
                self._transport_phase = _TransportPhase.PAYLOAD_DESCEND
                self._transport_started_at = t
                self._stall_since = None
                self._last_target = None
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_OPEN, GRIPPER_CLOSE))

        if self._transport_phase is _TransportPhase.PAYLOAD_DESCEND:
            assert self._transport_started_at is not None
            targets = (payload_descend, hold1)
            self._transport_watchdog(targets, eef0, eef1, t)
            steps_in_descend = t - self._transport_started_at
            reached = np.linalg.norm(targets[0] - eef0) <= self.config.position_tolerance
            is_stalled = (self._transport_phase is _TransportPhase.STALLED)
            
            if (
                reached
                or is_stalled
                or self._transport_pad_contact(0, "payload")
                or steps_in_descend >= self.PAYLOAD_DESCEND_HOLD_STEPS
            ):
                self._transport_phase = _TransportPhase.PAYLOAD_GRASP
                self._transport_started_at = t
                self._stall_since = None
                self._last_target = None
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_OPEN, GRIPPER_CLOSE))

        if self._transport_phase is _TransportPhase.PAYLOAD_GRASP:
            assert self._transport_started_at is not None
            held_for = t - self._transport_started_at
            if held_for >= self.config.grasp_hold_steps:
                payload_grasped = self._native_transport_grasp(0, "payload")
                trash_grasped = self._native_transport_grasp(1, "trash")
                if payload_grasped is not False and trash_grasped is not False:
                    self._payload_eef_offset = eef0 - payload
                    self._trash_eef_offset = eef1 - trash
                    self._transport_phase = _TransportPhase.PAYLOAD_LIFT
                    self._stall_since = None
                    self._last_target = None
                elif (
                    payload_grasped is False or trash_grasped is False
                ) and held_for >= self.config.grasp_hold_steps + self.GRASP_CONFIRM_STEPS:
                    self._stalled_from_phase = self._transport_phase.name
                    self._transport_phase = _TransportPhase.STALLED
            return self._two_arm_action(
                (eef0, eef1), eef0, eef1, (GRIPPER_CLOSE, GRIPPER_CLOSE)
            )

        lift_z = max(float(target_bin[2]), float(trash_bin[2])) + 0.20
        if self._transport_phase is _TransportPhase.PAYLOAD_LIFT:
            targets = (
                np.array([eef0[0], eef0[1], max(eef0[2], lift_z)]),
                np.array([eef1[0], eef1[1], max(eef1[2], lift_z)]),
            )
            self._transport_watchdog(targets, eef0, eef1, t)
            if min(eef0[2], eef1[2]) >= lift_z - self.config.position_scale:
                trash_target = trash_bin.copy()
                trash_target[2] += self.BIN_OBJECT_Z_OFFSET
                self._place_targets = (
                    np.array([0.0, -0.15, lift_z], dtype=np.float64),  # Arm0 Handover hover location
                    trash_target + self._trash_eef_offset,             # Arm1 goes to trash bin
                )
                self._transport_phase = _TransportPhase.TRASH_TRANSPORT
                self._stall_since = None
                self._last_target = None
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_CLOSE, GRIPPER_CLOSE))

        assert self._place_targets is not None
        
        if self._transport_phase is _TransportPhase.TRASH_TRANSPORT:
            targets = (
                self._place_targets[0],
                np.array([self._place_targets[1][0], self._place_targets[1][1], lift_z]),
            )
            self._transport_watchdog(targets, eef0, eef1, t)
            if self._both_reached(targets, eef0, eef1):
                self._transport_phase = _TransportPhase.TRASH_PLACE
                self._transport_started_at = t
                self._stall_since = None
                self._last_target = None
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_CLOSE, GRIPPER_CLOSE))

        if self._transport_phase is _TransportPhase.TRASH_PLACE:
            assert self._transport_started_at is not None
            # Arm0 waits. Arm1 descends strictly INTO the bin before opening the gripper.
            targets = self._place_targets
            self._transport_watchdog(targets, eef0, eef1, t)
            
            reached = self._both_reached(targets, eef0, eef1)
            is_stalled = (self._transport_phase is _TransportPhase.STALLED)
            
            if (reached or is_stalled) and t - self._transport_started_at >= self.config.place_hold_steps:
                self._transport_phase = _TransportPhase.HANDOVER_APPROACH
                # Freeze hammer head target instantly so we don't chase the pendulum swing!
                self._handover_head_target = self._head_center(payload, payload_quat).copy()
                self._stall_since = None
                self._last_target = None
                return self._two_arm_action(targets, eef0, eef1, (GRIPPER_CLOSE, GRIPPER_OPEN))
                
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_CLOSE, GRIPPER_CLOSE))

        if self._transport_phase is _TransportPhase.HANDOVER_APPROACH:
            assert self._handover_head_target is not None
            head = self._handover_head_target
            # Arm1 flies safely 15cm ABOVE the hammer head target to avoid horizontal sliding collision
            approach_z = max(lift_z, float(head[2])) + 0.15
            arm1_approach = np.array([head[0], head[1], approach_z], dtype=np.float64)
            targets = (self._place_targets[0], arm1_approach)
            self._transport_watchdog(targets, eef0, eef1, t)
            
            if np.linalg.norm(arm1_approach - eef1) <= self.config.position_tolerance:
                self._transport_phase = _TransportPhase.HANDOVER_DESCEND
                self._transport_started_at = t
                self._stall_since = None
                self._last_target = None
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_CLOSE, GRIPPER_OPEN))

        if self._transport_phase is _TransportPhase.HANDOVER_DESCEND:
            assert self._transport_started_at is not None
            assert self._handover_head_target is not None
            
            # Arm1 drops straight vertically down to pinch the frozen hammer head.
            arm1_descend = self._handover_head_target.copy()
            targets = (self._place_targets[0], arm1_descend)
            self._transport_watchdog(targets, eef0, eef1, t)
            
            steps_in_descend = t - self._transport_started_at
            reached = np.linalg.norm(arm1_descend - eef1) <= self.config.position_tolerance
            is_stalled = (self._transport_phase is _TransportPhase.STALLED)
            
            # A stall here means Arm1 made physical contact with the hammer head! Advance immediately!
            if reached or is_stalled or steps_in_descend >= 80:
                self._transport_phase = _TransportPhase.HANDOVER_GRASP
                self._transport_started_at = t
                self._stall_since = None
                self._last_target = None
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_CLOSE, GRIPPER_OPEN))

        if self._transport_phase is _TransportPhase.HANDOVER_GRASP:
            assert self._transport_started_at is not None
            assert self._handover_head_target is not None
            targets = (self._place_targets[0], self._handover_head_target)
            
            if t - self._transport_started_at >= self.config.grasp_hold_steps:
                self._transport_phase = _TransportPhase.HANDOVER_RELEASE
                self._transport_started_at = t
                self._stall_since = None
                self._last_target = None
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_CLOSE, GRIPPER_CLOSE))

        if self._transport_phase is _TransportPhase.HANDOVER_RELEASE:
            assert self._transport_started_at is not None
            # Arm0 OPENS, dropping the handle. Arm1 is now the sole owner of the payload.
            targets = (self._place_targets[0], eef1)
            if t - self._transport_started_at >= self.config.place_hold_steps:
                self._payload_eef_offset = eef1 - payload
                self._transport_phase = _TransportPhase.PAYLOAD_TRANSPORT
                self._stall_since = None
                self._last_target = None
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_OPEN, GRIPPER_CLOSE))

        if self._transport_phase is _TransportPhase.PAYLOAD_TRANSPORT:
            payload_target = target_bin.copy()
            payload_target[2] += self.BIN_OBJECT_Z_OFFSET
            arm1_transport = payload_target + self._payload_eef_offset
            
            # Arm1 flies high over the table before descending
            travel_z = max(lift_z, float(arm1_transport[2])) + 0.15
            arm1_high = np.array([arm1_transport[0], arm1_transport[1], travel_z])
            
            # Arm0 retracts out of the way
            arm0_retract = np.array([0.0, -0.3, travel_z])
            targets = (arm0_retract, arm1_high)
            
            self._transport_watchdog(targets, eef0, eef1, t)
            if self._both_reached(targets, eef0, eef1):
                self._transport_phase = _TransportPhase.PAYLOAD_PLACE
                self._transport_started_at = t
                self._place_targets = (arm0_retract, arm1_transport)
                self._stall_since = None
                self._last_target = None
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_OPEN, GRIPPER_CLOSE))

        if self._transport_phase is _TransportPhase.PAYLOAD_PLACE:
            assert self._transport_started_at is not None
            targets = self._place_targets
            self._transport_watchdog(targets, eef0, eef1, t)
            
            # Arm1 descends completely into the target bin before opening its gripper
            reached = self._both_reached(targets, eef0, eef1)
            is_stalled = (self._transport_phase is _TransportPhase.STALLED)
            
            if (reached or is_stalled) and t - self._transport_started_at >= self.config.place_hold_steps:
                self._transport_phase = _TransportPhase.DONE
                return self._two_arm_action(targets, eef0, eef1, (GRIPPER_OPEN, GRIPPER_OPEN))
                
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_OPEN, GRIPPER_CLOSE))

        return self._two_arm_action(
            (eef0, eef1), eef0, eef1, (GRIPPER_OPEN, GRIPPER_OPEN)
        )

    def _transport_watchdog(
        self,
        targets: tuple[np.ndarray, np.ndarray],
        eef0: np.ndarray,
        eef1: np.ndarray,
        t: int,
    ) -> None:
        """Two-arm variant of the base stall watchdog.

        Mirrors :meth:`ScriptedController._watchdog` but tracks the worst-case
        distance across both end-effectors and writes the ``STALLED`` state
        to the two-arm phase machine. Reuses the base reset fields
        (``_last_target``, ``_last_target_distance``, ``_stall_since``,
        ``_stalled_from_phase``) so :meth:`ScriptedController.reset` already
        clears them between episodes.
        """
        config = self.config
        target = np.concatenate(
            [np.asarray(t0, dtype=np.float64).ravel() for t0 in targets]
        )
        distance = float(
            max(
                np.linalg.norm(targets[0] - eef0),
                np.linalg.norm(targets[1] - eef1),
            )
        )
        if distance <= config.position_scale:
            self._stall_since = None
            self._last_target = target
            self._last_target_distance = distance
            return
        target_changed = self._last_target is None or not np.allclose(
            target, self._last_target, rtol=0.0, atol=1e-9
        )
        if target_changed or self._last_target_distance is None:
            self._last_target = target
            self._last_target_distance = distance
            self._stall_since = t
            return
        progress = self._last_target_distance - distance
        self._last_target_distance = distance
        if progress < config.stall_progress:
            if self._stall_since is None:
                self._stall_since = t
            elif t - self._stall_since >= config.stall_steps:
                self._stalled_from_phase = self._transport_phase.name
                self._transport_phase = _TransportPhase.STALLED
        else:
            self._stall_since = None

    def act(self, state: np.ndarray, t: int) -> np.ndarray:
        if self._two_arm:
            return self._transport_action(state, t)
        return super().act(state, t)

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
