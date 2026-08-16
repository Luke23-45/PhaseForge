"""Trace one or more state-only Transport oracle rollouts.

This diagnostic is deliberately separate from the rollout gate.  It never
changes the frozen reset bank, raw data, checkpoints, or evaluation results.
It records the facts needed to diagnose a two-arm Transport failure:

* both end-effector positions and action responses;
* the controller phase and exact phase targets;
* native grasp checks for the lid, payload, and trash;
* native body positions and state-vector object positions;
* EEF orientations, fingerpad / object collision-geom positions, and
  simulator contact pairs / distances;
* placement-target errors, gripper commands, and robosuite success.

Examples::

    uv run python scripts/debug_transport_rollout.py \
        --case-index 1 --max-steps 700 --log-every 10 \
        data=transport eval=rollout

    uv run python scripts/debug_transport_rollout.py \
        --cases 2 --max-steps 700 --log-every 25 \
        data=transport eval=rollout

The first command traces one known failing case from the reported gate.  The
second compares case 0 with case 1 without running the 50-case gate.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
from hydra import compose, initialize_config_module
from omegaconf import DictConfig

from phaseforge.evaluations.envs.task_registry import TaskSpec
from phaseforge.evaluations.rollout.runner import (
    _adapter_from_config,
    load_or_generate_bank,
    resolve_pinned_metadata,
    state_spec_from_config,
)
from phaseforge.utils.config import output_base_dir


def _jsonable(value: Any) -> Any:
    """Convert numpy/OmegaConf values into JSON-safe primitives."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _slice(state: np.ndarray, spec: Any, key: str) -> np.ndarray | None:
    """Return one declared state field, preserving schema errors as absence."""
    try:
        start, stop = spec.index_of(key)
    except (KeyError, ValueError):
        return None
    return np.asarray(state[start:stop], dtype=np.float64).copy()


def _body_positions(env: Any) -> dict[str, Any]:
    """Read Transport body positions without assuming one robosuite layout."""
    result: dict[str, Any] = {}
    transport = getattr(env, "transport", None)
    objects = getattr(transport, "objects", None)
    if not isinstance(objects, dict):
        objects = {}

    body_ids = getattr(env, "obj_body_id", {})
    if not isinstance(body_ids, dict):
        body_ids = {}

    for name in ("lid", "payload", "trash"):
        body_id = body_ids.get(name)
        obj = objects.get(name)
        body_name = getattr(obj, "root_body", None) or getattr(obj, "name", None)
        try:
            if body_id is None and body_name is not None:
                body_id = env.sim.model.body_name2id(str(body_name))
            if body_id is not None:
                result[name] = np.asarray(
                    env.sim.data.body_xpos[int(body_id)], dtype=np.float64
                ).copy()
        except Exception as exc:  # noqa: BLE001 - diagnostics must continue
            result[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _gripper_qpos(env: Any) -> dict[str, Any]:
    """Read native gripper joint positions for both robots when available."""
    result: dict[str, Any] = {}
    for robot_index, robot in enumerate(getattr(env, "robots", ())):
        try:
            arms = tuple(getattr(robot, "arms", ()))
            arm = arms[0] if arms else "right"
            indexes = getattr(robot, "_ref_gripper_joint_pos_indexes")
            if isinstance(indexes, dict):
                indexes = indexes[arm]
            result[f"robot{robot_index}"] = np.asarray(
                env.sim.data.qpos[indexes], dtype=np.float64
            ).copy()
        except Exception as exc:  # noqa: BLE001 - diagnostics must continue
            result[f"robot{robot_index}_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _native_grasps(controller: Any) -> dict[str, Any]:
    """Call the same privileged grasp checks used by the Transport oracle."""
    checks: dict[str, Any] = {}
    for arm_index, object_name in (
        (0, "lid"),
        (1, "trash"),
        (0, "payload"),
        (1, "payload"),
    ):
        key = f"arm{arm_index}_{object_name}"
        try:
            checks[key] = controller._native_transport_grasp(arm_index, object_name)
        except Exception as exc:  # noqa: BLE001 - diagnostics must continue
            checks[key] = f"{type(exc).__name__}: {exc}"
    return checks


def _pad_contacts(controller: Any) -> dict[str, Any]:
    """Fingerpad contact probes used to confirm the payload descent reached the hammer."""
    probes: dict[str, Any] = {}
    for arm_index, object_name in (
        (0, "payload"),
        (0, "lid"),
        (1, "trash"),
        (1, "payload"),
    ):
        key = f"arm{arm_index}_pad_{object_name}"
        try:
            probes[key] = controller._transport_pad_contact(arm_index, object_name)
        except Exception as exc:  # noqa: BLE001 - diagnostics must continue
            probes[key] = f"{type(exc).__name__}: {exc}"
    return probes


def _pad_contact_sides(controller: Any) -> dict[str, Any]:
    """Report each fingerpad separately for the two unresolved grasp probes."""
    details: dict[str, Any] = {}
    for arm_index, object_name in ((1, "trash"), (1, "payload")):
        key = f"arm{arm_index}_pad_{object_name}_sides"
        try:
            env = controller.env
            checker = getattr(env, "check_contact")
            robot = env.robots[arm_index]
            objects = env.transport.objects
            mapping = robot.gripper
            gripper = mapping.get(robot.arms[0]) if isinstance(mapping, dict) else mapping
            important = gripper.important_geoms
            object_geoms = getattr(objects[object_name], "contact_geoms")
            details[key] = {
                side: bool(checker(important[geom], object_geoms))
                for side, geom in (
                    ("left", "left_fingerpad"),
                    ("right", "right_fingerpad"),
                )
            }
        except Exception as exc:  # noqa: BLE001 - diagnostics must continue
            details[key] = f"{type(exc).__name__}: {exc}"
    return details


def _geom_names(value: Any) -> list[str]:
    """Normalize robosuite's string-or-list geom declarations."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _geom_id(model: Any, name: str) -> int | None:
    """Resolve a MuJoCo geom id across mujoco-py and mujoco bindings."""
    try:
        resolver = getattr(model, "geom_name2id", None)
        if callable(resolver):
            return int(resolver(name))
    except Exception:  # noqa: BLE001 - diagnostics must continue
        pass
    try:
        geom = model.geom(name)
        geom_id = getattr(geom, "id", None)
        return None if geom_id is None else int(geom_id)
    except Exception:  # noqa: BLE001 - diagnostics must continue
        return None


def _geom_id_with_error(model: Any, name: str) -> tuple[int | None, str | None]:
    """Resolve a geom id and retain the exact failure for diagnostics."""
    resolver = getattr(model, "geom_name2id", None)
    if callable(resolver):
        try:
            return int(resolver(name)), None
        except Exception as exc:  # noqa: BLE001 - diagnostics must continue
            first_error = f"geom_name2id: {type(exc).__name__}: {exc}"
    else:
        first_error = "geom_name2id unavailable"
    try:
        geom = model.geom(name)
        geom_id = getattr(geom, "id", None)
        if geom_id is not None:
            return int(geom_id), None
        return None, f"model.geom returned no id for {name!r}"
    except Exception as exc:  # noqa: BLE001 - diagnostics must continue
        return None, f"{first_error}; model.geom: {type(exc).__name__}: {exc}"


def _geom_name(model: Any, geom_id: int) -> str:
    """Resolve a MuJoCo geom name without making the probe version-specific."""
    for method_name in ("geom_id2name",):
        try:
            resolver = getattr(model, method_name, None)
            if callable(resolver):
                value = resolver(int(geom_id))
                if value is not None:
                    return str(value)
        except Exception:  # noqa: BLE001 - diagnostics must continue
            pass
    try:
        resolver = getattr(model, "id2name", None)
        if callable(resolver):
            value = resolver(int(geom_id), "geom")
            if value is not None:
                return str(value)
    except Exception:  # noqa: BLE001 - diagnostics must continue
        pass
    return str(geom_id)


def _contact_geometry(controller: Any) -> dict[str, Any]:
    """Capture exact pad/object geom positions and active contact pairs.

    A boolean ``check_contact`` result cannot explain whether a target is
    wrong, a pad is on the wrong side of the object, or contact is being lost
    during release. This probe records the underlying geom-level evidence.
    """
    result: dict[str, Any] = {}
    env = getattr(controller, "env", None)
    try:
        model = env.sim.model
        data = env.sim.data
        robots = env.robots
        objects = env.transport.objects
    except Exception as exc:  # noqa: BLE001 - diagnostics must continue
        return {"error": f"{type(exc).__name__}: {exc}"}

    result["simulator"] = {
        "model_type": type(model).__name__,
        "data_type": type(data).__name__,
        "model_ngeom": getattr(model, "ngeom", None),
        "has_geom_name2id": callable(getattr(model, "geom_name2id", None)),
        "has_geom": callable(getattr(model, "geom", None)),
        "has_geom_id2name": callable(getattr(model, "geom_id2name", None)),
        "has_id2name": callable(getattr(model, "id2name", None)),
        "has_geom_xpos": hasattr(data, "geom_xpos"),
        "ncon": int(getattr(data, "ncon", 0)),
    }

    for arm_index, object_name in (
        (0, "lid"),
        (1, "trash"),
        (0, "payload"),
        (1, "payload"),
    ):
        key = f"arm{arm_index}_{object_name}"
        try:
            robot = robots[arm_index]
            mapping = robot.gripper
            gripper = (
                mapping.get(robot.arms[0])
                if isinstance(mapping, dict)
                else mapping
            )
            important = gripper.important_geoms
            pad_names = {
                side: _geom_names(important[group])
                for side, group in (
                    ("left", "left_fingerpad"),
                    ("right", "right_fingerpad"),
                )
            }
            object_names = _geom_names(objects[object_name].contact_geoms)
            pad_ids: dict[str, dict[str, int | None]] = {}
            object_ids: dict[str, int | None] = {}
            resolution_errors: dict[str, str] = {}
            for side, names in pad_names.items():
                pad_ids[side] = {}
                for name in names:
                    geom_id, error = _geom_id_with_error(model, name)
                    pad_ids[side][name] = geom_id
                    if error is not None:
                        resolution_errors[f"pad:{side}:{name}"] = error
            for name in object_names:
                geom_id, error = _geom_id_with_error(model, name)
                object_ids[name] = geom_id
                if error is not None:
                    resolution_errors[f"object:{name}"] = error

            def geom_position(geom_id: int | None) -> np.ndarray | None:
                if geom_id is None:
                    return None
                try:
                    return np.asarray(data.geom_xpos[geom_id], dtype=np.float64).copy()
                except Exception as exc:  # noqa: BLE001 - diagnostics must continue
                    resolution_errors[f"position:{geom_id}"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    return None

            pad_positions = {
                side: {name: geom_position(geom_id) for name, geom_id in ids.items()}
                for side, ids in pad_ids.items()
            }
            object_positions = {
                name: geom_position(geom_id) for name, geom_id in object_ids.items()
            }
            valid_object_positions = [
                pos for pos in object_positions.values() if isinstance(pos, np.ndarray)
            ]
            min_distances: dict[str, float | None] = {}
            for side, side_positions in pad_positions.items():
                valid_pad_positions = [
                    pos for pos in side_positions.values() if isinstance(pos, np.ndarray)
                ]
                min_distances[side] = (
                    None
                    if not valid_pad_positions or not valid_object_positions
                    else float(
                        min(
                            np.linalg.norm(pad_pos - object_pos)
                            for pad_pos in valid_pad_positions
                            for object_pos in valid_object_positions
                        )
                    )
                )

            pad_id_set = {
                geom_id
                for side_ids in pad_ids.values()
                for geom_id in side_ids.values()
                if geom_id is not None
            }
            object_id_set = {geom_id for geom_id in object_ids.values() if geom_id is not None}
            contacts: list[dict[str, Any]] = []
            for contact_index in range(int(getattr(data, "ncon", 0))):
                contact = data.contact[contact_index]
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                if (
                    (geom1 in pad_id_set and geom2 in object_id_set)
                    or (geom2 in pad_id_set and geom1 in object_id_set)
                ):
                    contacts.append(
                        {
                            "geom1": _geom_name(model, geom1),
                            "geom2": _geom_name(model, geom2),
                            "dist": float(contact.dist),
                        }
                    )
            result[key] = {
                "pad_names": pad_names,
                "object_names": object_names,
                "pad_ids": pad_ids,
                "object_ids": object_ids,
                "resolution_errors": resolution_errors,
                "pad_positions": pad_positions,
                "object_positions": object_positions,
                "min_pad_object_distance": min_distances,
                "contacts": contacts,
            }
        except Exception as exc:  # noqa: BLE001 - diagnostics must continue
            result[key] = f"{type(exc).__name__}: {exc}"
    return result


def _phase_targets(
    controller: Any, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Reconstruct the controller's two-arm target for the current phase."""
    (
        payload,
        payload_quat,
        trash,
        lid_handle,
        target_bin,
        trash_bin,
    ) = controller._transport_values(state)
    eef0 = controller.eef_pos(state)
    eef1 = controller._eef1_pos(state)
    config = controller.config
    phase = controller.phase_name

    if controller._lid_clear_target is None:
        controller._lid_clear_target = lid_handle + controller.LID_CLEAR_OFFSET

    lid_approach = lid_handle + np.array([0.0, 0.0, config.approach_z_offset])
    lid_grasp = lid_handle.copy()
    trash_high = np.array(
        [
            trash[0],
            trash[1],
            min(
                trash[2] + controller.TRASH_APPROACH_Z_OFFSET,
                controller.TRASH_APPROACH_Z_CAP,
            ),
        ],
        dtype=np.float64,
    )
    trash_grasp = trash + np.array([0.0, 0.0, controller.TRASH_GRASP_Z_OFFSET])

    if phase == "LID_APPROACH":
        return lid_approach, trash_high
    if phase == "LID_DESCEND":
        return lid_grasp, trash_grasp
    if phase == "LID_GRASP":
        return eef0.copy(), eef1.copy()
    if phase == "LID_LIFT":
        return np.array([eef0[0], eef0[1], max(eef0[2], 1.08)]), eef1.copy()
    if phase == "LID_CLEAR":
        return controller._lid_clear_target.copy(), eef1.copy()

    payload_approach_z = max(
        float(payload[2]) + config.approach_z_offset, 1.08
    )
    payload_approach = np.array(
        [payload[0], payload[1], payload_approach_z], dtype=np.float64
    )
    payload_descend = payload + np.array([0.0, 0.0, 0.01], dtype=np.float64)
    if phase == "PAYLOAD_APPROACH":
        return payload_approach, eef1.copy()
    if phase == "PAYLOAD_DESCEND":
        return payload_descend, eef1.copy()
    if phase == "PAYLOAD_GRASP":
        return eef0.copy(), eef1.copy()

    lift_z = max(float(target_bin[2]), float(trash_bin[2])) + 0.20
    if phase == "PAYLOAD_LIFT":
        return (
            np.array([eef0[0], eef0[1], max(eef0[2], lift_z)]),
            np.array([eef1[0], eef1[1], max(eef1[2], lift_z)]),
        )

    if phase == "TRASH_TRANSPORT":
        if controller._place_targets is None:
            return None
        return (
            controller._place_targets[0],
            np.array(
                [
                    controller._place_targets[1][0],
                    controller._place_targets[1][1],
                    lift_z,
                ]
            ),
        )
    if phase == "TRASH_PLACE":
        return controller._place_targets
    if phase == "TRASH_RELEASE":
        # Release keeps both arms at the stored placement targets for the
        # same 15-step hold used by _transport_action().
        return controller._place_targets

    # Mid-air handover in the 0.6m gap between the two tables (no shared
    # table surface exists in MultiTableArena). The meeting point mirrors
    # the controller's live handle-axis target and therefore remains in
    # the same frame as the demonstrated arm1 payload grasp.
    meeting = controller._payload_meeting_point(payload, payload_quat, eef1)

    if phase == "TABLE_TRANSPORT":
        # Keep this identical to _transport_action(): arm0 holds its live
        # end-effector position while arm1 converges on the meeting point.
        # Reporting the meeting point for arm0 here made the diagnostic show
        # a false ~9 cm arm0 error even when the controller commanded zero
        # position error for that arm.
        return eef0.copy(), meeting.copy()
    if phase == "TABLE_DESCEND":
        frozen = getattr(controller, "_handover_meeting_target", None)
        grasp_target = getattr(controller, "_handover_arm1_grasp_target", None)
        return eef0.copy(), (
            grasp_target.copy()
            if grasp_target is not None
            else (meeting if frozen is None else frozen.copy())
        )
    if phase == "TABLE_RELEASE":
        # The production controller holds arm0 at its current EEF pose while
        # arm1 holds the first measured native-contact pose.
        frozen = getattr(controller, "_handover_meeting_target", None)
        grasp_target = getattr(controller, "_handover_arm1_grasp_target", None)
        return eef0.copy(), (
            grasp_target.copy()
            if grasp_target is not None
            else (meeting if frozen is None else frozen.copy())
        )

    # After TABLE_RELEASE, the controller snapshots the meeting point so arm1 holds steady
    snapshot = getattr(controller, "_handover_arm1_snapshot", meeting)
    if snapshot is None:
        snapshot = meeting

    if phase == "TABLE_RETRACT":
        return np.array([0.0, -0.40, lift_z]), snapshot.copy()
    if phase in ("HANDOVER_APPROACH", "HANDOVER_DESCEND", "HANDOVER_GRASP"):
        return np.array([0.0, -0.40, lift_z]), snapshot.copy()
    if phase == "HANDOVER_LIFT":
        return np.array([0.0, -0.40, lift_z]), np.array(
            [snapshot[0], snapshot[1], lift_z]
        )
    if phase == "HANDOVER_SWING":
        return np.array([0.0, -0.40, lift_z]), np.array(
            [float(target_bin[0]), snapshot[1], lift_z]
        )

    if phase == "PAYLOAD_TRANSPORT":
        payload_target = target_bin.copy()
        payload_target[2] += controller.BIN_OBJECT_Z_OFFSET
        arm1_transport = payload_target + controller._payload_eef_offset
        arm1_high = np.array([arm1_transport[0], arm1_transport[1], lift_z])
        arm0_retract = np.array([0.0, -0.4, lift_z])
        return arm0_retract, arm1_high
    if phase == "PAYLOAD_PLACE":
        return controller._place_targets
    if phase == "PAYLOAD_RETRACT":
        return (
            controller._place_targets[0],
            np.array([eef1[0], eef1[1], lift_z]),
        )
    return None


def _quat_xyzw_to_mat(value: Any) -> np.ndarray | None:
    """Convert a normalized-or-unnormalized robosuite ``xyzw`` quaternion.

    The state schema exposes robosuite quaternions in ``(x, y, z, w)`` order.
    The explicit normalization makes this diagnostic safe if a simulator
    snapshot contains a small floating-point norm error.
    """
    try:
        quat = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if quat.size != 4 or not np.all(np.isfinite(quat)):
        return None
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return None
    x, y, z, w = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    """Return the principal angle of a 3D rotation matrix in degrees."""
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _orientation_delta_deg(before: Any, after: Any) -> float | None:
    """Return a safe world-frame EEF orientation impulse magnitude."""
    before_rotation = _quat_xyzw_to_mat(before)
    after_rotation = _quat_xyzw_to_mat(after)
    if before_rotation is None or after_rotation is None:
        return None
    return _rotation_angle_deg(before_rotation.T @ after_rotation)


def _handover_orientation_diagnostics(
    snapshot: dict[str, Any],
    baseline: dict[str, np.ndarray | int] | None,
    step_index: int,
) -> tuple[dict[str, np.ndarray | int] | None, dict[str, Any] | None]:
    """Measure arm-1 orientation and grasp offset in the payload frame.

    ``R_payload.T @ R_eef1`` is the arm-1 orientation expressed in the
    payload frame.  The baseline is captured at the first native arm-1
    payload grasp during the handover, then every later sample is compared
    with that baseline.  This distinguishes a moving payload target from a
    changing relative pose; it is diagnostic only and does not affect the
    controller action.
    """
    if snapshot.get("phase") not in {"TABLE_DESCEND", "TABLE_RELEASE"}:
        return baseline, None

    payload_rotation = _quat_xyzw_to_mat(snapshot.get("state_payload_quat"))
    eef_rotation = _quat_xyzw_to_mat(snapshot.get("eef1_quat"))
    payload = snapshot.get("state_payload")
    eef1 = snapshot.get("eef1")
    try:
        payload_position = np.asarray(payload, dtype=np.float64).reshape(-1)
        eef1_position = np.asarray(eef1, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return baseline, {"error": "invalid position snapshot"}
    if (
        payload_rotation is None
        or eef_rotation is None
        or payload_position.size != 3
        or eef1_position.size != 3
    ):
        return baseline, {"error": "invalid quaternion or position snapshot"}

    relative_rotation = payload_rotation.T @ eef_rotation
    payload_frame_offset = payload_rotation.T @ (eef1_position - payload_position)
    native_arm1 = snapshot.get("native_grasps", {}).get("arm1_payload")
    if baseline is None and native_arm1 is True:
        baseline = {
            "step": int(step_index),
            "relative_rotation": relative_rotation.copy(),
            "payload_frame_offset": payload_frame_offset.copy(),
        }

    result: dict[str, Any] = {
        "baseline_step": None if baseline is None else baseline["step"],
        "payload_frame_eef_offset": payload_frame_offset,
        "payload_frame_offset_error_m": None,
        "relative_orientation_change_deg": None,
    }
    if baseline is not None:
        result["payload_frame_offset_error_m"] = float(
            np.linalg.norm(payload_frame_offset - baseline["payload_frame_offset"])
        )
        result["relative_orientation_change_deg"] = _rotation_angle_deg(
            baseline["relative_rotation"].T @ relative_rotation
        )
    return baseline, result


def _snapshot(controller: Any, state: np.ndarray, env: Any) -> dict[str, Any]:
    """Capture all geometry and controller state at one timestep."""
    eef0 = controller.eef_pos(state)
    eef1 = controller._eef1_pos(state)
    (
        payload,
        payload_quat,
        trash,
        lid_handle,
        target_bin,
        trash_bin,
    ) = controller._transport_values(state)
    targets = _phase_targets(controller, state)
    distances = None
    if targets is not None:
        distances = {
            "arm0": float(np.linalg.norm(targets[0] - eef0)),
            "arm1": float(np.linalg.norm(targets[1] - eef1)),
        }
    body_positions = _body_positions(env)
    body_target_distances: dict[str, float] = {}
    for body_name, target in (("payload", target_bin), ("trash", trash_bin)):
        body = body_positions.get(body_name)
        if isinstance(body, np.ndarray):
            body_target_distances[body_name] = float(np.linalg.norm(body - target))
    try:
        env_success = bool(env._check_success())
    except Exception as exc:  # noqa: BLE001 - diagnostics must continue
        env_success = f"{type(exc).__name__}: {exc}"
    return {
        "phase": controller.phase_name,
        "stalled_from_phase": controller.stalled_from_phase,
        "eef0": eef0,
        "eef1": eef1,
        "eef0_quat": _slice(state, controller.state_spec, "robot0_eef_quat"),
        "eef1_quat": _slice(state, controller.state_spec, "robot1_eef_quat"),
        "state_payload": payload,
        "state_payload_quat": payload_quat,
        "state_trash": trash,
        "state_lid_handle": lid_handle,
        "state_target_bin": target_bin,
        "state_trash_bin": trash_bin,
        "phase_targets": targets,
        "phase_target_distance": distances,
        "place_targets": controller._place_targets,
        "payload_eef_offset": controller._payload_eef_offset,
        "trash_eef_offset": controller._trash_eef_offset,
        "transport_started_at": controller._transport_started_at,
        "handover_native_stable_steps": getattr(
            controller, "_handover_native_stable_steps", None
        ),
        "handover_arm1_grasp_target": getattr(
            controller, "_handover_arm1_grasp_target", None
        ),
        "handover_arm1_grasp_offset_local": getattr(
            controller, "_handover_arm1_grasp_offset_local", None
        ),
        "native_grasps": _native_grasps(controller),
        "pad_contacts": _pad_contacts(controller),
        "pad_contact_sides": _pad_contact_sides(controller),
        "contact_geometry": _contact_geometry(controller),
        "body_positions": body_positions,
        "body_target_distance": body_target_distances,
        "gripper_qpos": _gripper_qpos(env),
        "env_success": env_success,
    }


def _contact_geometry_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Compact geom-level summary suitable for the human-readable log."""
    summary: dict[str, Any] = {}
    for key, details in snapshot.get("contact_geometry", {}).items():
        if not isinstance(details, dict):
            summary[key] = details
            continue
        summary[key] = {
            "min_dist": details.get("min_pad_object_distance"),
            "contacts": len(details.get("contacts", [])),
            "pad_ids": details.get("pad_ids"),
            "object_ids": details.get("object_ids"),
            "resolution_errors": details.get("resolution_errors"),
        }
    return summary


def _handover_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return compact per-step geometry evidence for the active handover."""
    result: dict[str, Any] = {}
    geometry = snapshot.get("contact_geometry", {})
    for key in ("arm0_payload", "arm1_payload"):
        details = geometry.get(key)
        if not isinstance(details, dict):
            result[key] = details
            continue
        result[key] = {
            "min_dist": details.get("min_pad_object_distance"),
            "contacts": details.get("contacts", []),
        }
    return result


def _probe_action_response(adapter: Any, case: Any, spec: Any) -> list[dict[str, Any]]:
    """Measure position and orientation action response for both arms.

    The probe resets before every impulse, so the reported response is not
    contaminated by the preceding impulse.  Orientation response is reported
    as the principal angle between the pre/post EEF rotations, while the raw
    quaternions remain available for checking sign and axis conventions.
    """
    result: list[dict[str, Any]] = []
    for arm_offset, arm_name in ((0, "arm0"), (7, "arm1")):
        for kind, axis_offset, axis_names in (
            ("position", 0, ("x", "y", "z")),
            ("orientation", 3, ("roll", "pitch", "yaw")),
        ):
            for axis, axis_name in enumerate(axis_names):
                state = adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
                before0 = _slice(state, spec, "robot0_eef_pos")
                before1 = _slice(state, spec, "robot1_eef_pos")
                before0_quat = _slice(state, spec, "robot0_eef_quat")
                before1_quat = _slice(state, spec, "robot1_eef_quat")
                action = np.zeros(adapter.action_dim, dtype=np.float64)
                action[arm_offset + axis_offset + axis] = 0.25
                action[arm_offset + 6] = -1.0
                after, _done, success, _info = adapter.step(action)
                after0 = _slice(after, spec, "robot0_eef_pos")
                after1 = _slice(after, spec, "robot1_eef_pos")
                after0_quat = _slice(after, spec, "robot0_eef_quat")
                after1_quat = _slice(after, spec, "robot1_eef_quat")
                result.append(
                    {
                        "arm": arm_name,
                        "kind": kind,
                        "axis": axis_name,
                        "action": action,
                        "eef0_delta": (
                            None
                            if before0 is None or after0 is None
                            else after0 - before0
                        ),
                        "eef1_delta": (
                            None
                            if before1 is None or after1 is None
                            else after1 - before1
                        ),
                        "eef0_quat_before": before0_quat,
                        "eef0_quat_after": after0_quat,
                        "eef1_quat_before": before1_quat,
                        "eef1_quat_after": after1_quat,
                        "eef0_orientation_delta_deg": _orientation_delta_deg(
                            before0_quat, after0_quat
                        ),
                        "eef1_orientation_delta_deg": _orientation_delta_deg(
                            before1_quat, after1_quat
                        ),
                        "success_after_step": bool(success),
                    }
                )
    return result


def _trace_case(
    adapter: Any,
    case: Any,
    spec: Any,
    controller_cls: type,
    *,
    max_steps: int,
    log_every: int,
    include_probe: bool,
) -> dict[str, Any]:
    controller = controller_cls(spec, env=adapter.env)
    state = adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
    record: dict[str, Any] = {
        "case_index": int(case.index),
        "max_steps": max_steps,
        "action_response_probe": (
            _probe_action_response(adapter, case, spec) if include_probe else []
        ),
        "steps": [],
        "success": False,
        "termination": "horizon",
    }
    state = adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
    handover_orientation_baseline: dict[str, np.ndarray | int] | None = None

    for t in range(max_steps):
        phase_before = controller.phase_name
        before = _snapshot(controller, state, adapter.env)
        action = np.asarray(controller.act(state, t), dtype=np.float64)
        command = {
            "arm0_xyz": action[:3].copy(),
            "arm0_orientation": action[3:6].copy(),
            "arm0_gripper": float(action[6]),
            "arm1_xyz": action[7:10].copy(),
            "arm1_orientation": action[10:13].copy(),
            "arm1_gripper": float(action[13]),
        }
        phase_after_action = controller.phase_name
        next_state, _done, success, info = adapter.step(action)
        after = _snapshot(controller, next_state, adapter.env)
        handover_orientation_baseline, orientation_metrics = (
            _handover_orientation_diagnostics(
                after, handover_orientation_baseline, t
            )
        )
        if orientation_metrics is not None:
            after["handover_orientation"] = orientation_metrics
        step = {
            "t": t,
            "phase_before": phase_before,
            "phase_after_action": phase_after_action,
            "phase_after_step": controller.phase_name,
            "command": command,
            "before": before,
            "after": after,
            "eef0_delta": after["eef0"] - before["eef0"],
            "eef1_delta": after["eef1"] - before["eef1"],
            "adapter_info": info,
            "success": bool(success),
        }
        record["steps"].append(step)

        if (
            t % max(1, log_every) == 0
            or phase_before != controller.phase_name
            or success
            or controller.phase_name == "STALLED"
        ):
            distances = after["phase_target_distance"] or {}
            eef0 = after["eef0"]
            payload = after["state_payload"]
            xy_err = float(np.linalg.norm(eef0[:2] - payload[:2]))
            print(
                f"case={case.index:02d} t={t:03d} "
                f"phase={phase_before}->{controller.phase_name} "
                f"d0={distances.get('arm0', float('nan')):.4f} "
                f"d1={distances.get('arm1', float('nan')):.4f} "
                f"xy_e0_p={xy_err:.4f} "
                f"g={after['native_grasps']} "
                f"p={after.get('pad_contacts')} "
                f"ps={after.get('pad_contact_sides')} "
                f"stable={after.get('handover_native_stable_steps')} "
                f"bd={after.get('body_target_distance')} "
                f"env={after.get('env_success')} "
                f"success={bool(success)}"
            )
            if (
                phase_before != controller.phase_name
                or success
                or controller.phase_name == "STALLED"
            ):
                print(
                    f"  detail case={case.index:02d} t={t:03d} "
                    f"eef0={np.round(after['eef0'], 4).tolist()} "
                    f"eef1={np.round(after['eef1'], 4).tolist()} "
                    f"q0={_jsonable(after.get('eef0_quat'))} "
                    f"q1={_jsonable(after.get('eef1_quat'))} "
                    f"payload={np.round(after['state_payload'], 4).tolist()} "
                    f"trash={np.round(after['state_trash'], 4).tolist()} "
                    f"targets={_jsonable(after.get('phase_targets'))} "
                    f"geom={_contact_geometry_summary(after)}"
                )

        if phase_before in {"TABLE_DESCEND", "TABLE_RELEASE"} or controller.phase_name in {
            "TABLE_DESCEND",
            "TABLE_RELEASE",
        }:
            print(
                f"HANDOVER case={case.index:02d} t={t:03d} "
                f"phase={phase_before}->{controller.phase_name} "
                f"stable={after.get('handover_native_stable_steps')} "
                f"native={after.get('native_grasps')} "
                f"pads={after.get('pad_contact_sides')} "
                f"eef0={np.round(after['eef0'], 5).tolist()} "
                f"eef1={np.round(after['eef1'], 5).tolist()} "
                f"eef1_quat={_jsonable(after.get('eef1_quat'))} "
                f"payload_quat={_jsonable(after.get('state_payload_quat'))} "
                f"orientation={_jsonable(after.get('handover_orientation'))} "
                f"payload={np.round(after['state_payload'][:3], 5).tolist()} "
                f"targets={_jsonable(after.get('phase_targets'))} "
                f"command={_jsonable(command)} "
                f"body={_jsonable(after.get('body_positions', {}).get('payload'))} "
                f"geom={_handover_summary(after)}"
            )

        state = next_state
        if success:
            record["success"] = True
            record["termination"] = "success"
            break
        if controller.phase_name == "STALLED":
            record["termination"] = "stalled"
            break

    record["final_phase"] = controller.phase_name
    record["stalled_from_phase"] = controller.stalled_from_phase
    if record["steps"]:
        record["final_snapshot"] = record["steps"][-1]["after"]
        final = record["steps"][-1]["after"]
        print(
            f"FINAL case={case.index:02d} phase={controller.phase_name} "
            f"success={record['success']} env={final.get('env_success')} "
            f"body_dist={final.get('body_target_distance')} "
            f"native={final.get('native_grasps')} "
            f"contacts={_contact_geometry_summary(final)}"
        )
    return _jsonable(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1, help="trace bank cases 0..N-1")
    parser.add_argument(
        "--case-index",
        type=int,
        action="append",
        help="trace an exact bank case; repeat for multiple cases",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-action-probe", action="store_true")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, for example data=transport eval=rollout",
    )
    args = parser.parse_args()
    if args.cases <= 0 or args.log_every <= 0:
        parser.error("--cases and --log-every must be positive")
    if args.case_index is not None and any(index < 0 for index in args.case_index):
        parser.error("--case-index values must be non-negative")

    overrides = args.overrides or ["data=transport", "eval=rollout"]
    with initialize_config_module(version_base="1.3", config_module="phaseforge.config"):
        cfg: DictConfig = compose(config_name="main", overrides=overrides)

    meta = resolve_pinned_metadata(cfg)
    spec = state_spec_from_config(cfg)
    adapter = _adapter_from_config(cfg, meta)
    traces: list[dict[str, Any]] = []
    try:
        bank = load_or_generate_bank(cfg, meta)
        if bank.task != "Transport":
            raise ValueError(
                f"This diagnostic requires a Transport bank, got {bank.task!r}. "
                "Use data=transport and the matching raw HDF5."
            )
        controller_cls = TaskSpec.from_protocol(bank.task).get_controller_class()
        if args.case_index is not None:
            invalid = [index for index in args.case_index if index >= bank.num_cases]
            if invalid:
                raise ValueError(
                    f"case index {invalid} outside bank range 0..{bank.num_cases - 1}"
                )
            selected = [bank.cases[index] for index in args.case_index]
        else:
            selected = bank.cases[: args.cases]
        if not selected:
            raise ValueError("No cases selected")
        max_steps = int(
            adapter.horizon if args.max_steps is None else args.max_steps
        )
        # The bank may be pinned to a shorter horizon than the protocol default
        # (Transport: protocol=700, some banks=500). Allow the user to run the
        # full protocol horizon regardless of the bank pin, since the adapter
        # does not enforce an internal step limit (``done`` is always False).
        protocol_horizon = TaskSpec.from_protocol(bank.task).horizon
        if max_steps <= 0 or max_steps > protocol_horizon:
            raise ValueError(
                f"--max-steps must be in 1..{protocol_horizon}, got {max_steps}"
            )

        print(
            f"Task: {bank.task}; bank: {bank.bank_id}; "
            f"cases: {[case.index for case in selected]}; "
            f"adapter_horizon: {adapter.horizon}; trace_max_steps: {max_steps}"
        )
        for case in selected:
            traces.append(
                _trace_case(
                    adapter,
                    case,
                    spec,
                    controller_cls,
                    max_steps=max_steps,
                    log_every=args.log_every,
                    include_probe=not args.no_action_probe,
                )
            )
    finally:
        adapter.close()

    output_dir = output_base_dir(cfg) / "_gates"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"debug_transport_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    payload = {
        "task": "Transport",
        "bank_id": bank.bank_id,
        "config_overrides": overrides,
        "env_name": meta.env_name,
        "env_version": meta.env_version,
        "state_dim": spec.dim,
        "action_dim": int(cfg.data.action_dim),
        "traces": traces,
    }
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Complete Transport diagnostic written to {path}")


if __name__ == "__main__":
    main()
