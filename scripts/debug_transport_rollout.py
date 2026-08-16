"""Trace one or more state-only Transport oracle rollouts.

This diagnostic is deliberately separate from the rollout gate.  It never
changes the frozen reset bank, raw data, checkpoints, or evaluation results.
It records the facts needed to diagnose a two-arm Transport failure:

* both end-effector positions and action responses;
* the controller phase and exact phase targets;
* native grasp checks for the lid, payload, and trash;
* native body positions and state-vector object positions;
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
    for arm_index, object_name in ((0, "payload"), (0, "lid"), (1, "trash")):
        key = f"arm{arm_index}_pad_{object_name}"
        try:
            probes[key] = controller._transport_pad_contact(arm_index, object_name)
        except Exception as exc:  # noqa: BLE001 - diagnostics must continue
            probes[key] = f"{type(exc).__name__}: {exc}"
    return probes


def _phase_targets(
    controller: Any, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Reconstruct the controller's two-arm target for the current phase."""
    payload, trash, lid_handle, target_bin, trash_bin = controller._transport_values(state)
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

    payload_approach = payload + np.array([0.0, 0.0, config.approach_z_offset])
    state_obj = state[controller.obj_start : controller.obj_end]
    payload_quat = state_obj[3:7]
    head_offset = controller._quat_xyzw_to_mat(payload_quat) @ np.array(
        [0.0, 0.0, controller.PAYLOAD_HEAD_OFFSET_Z], dtype=np.float64
    )
    head_center = payload + head_offset
    payload_descend = head_center + np.array(
        [0.0, 0.0, controller.PAYLOAD_HEAD_DESCEND_Z_OFFSET]
    )
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

    if phase == "TRANSPORT":
        if controller._place_targets is None:
            return None
        return (
            np.array([controller._place_targets[0][0], controller._place_targets[0][1], lift_z]),
            np.array([controller._place_targets[1][0], controller._place_targets[1][1], lift_z]),
        )
    if phase == "PLACE":
        return controller._place_targets
    return None


def _snapshot(controller: Any, state: np.ndarray, env: Any) -> dict[str, Any]:
    """Capture all geometry and controller state at one timestep."""
    eef0 = controller.eef_pos(state)
    eef1 = controller._eef1_pos(state)
    payload, trash, lid_handle, target_bin, trash_bin = controller._transport_values(state)
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
        "state_payload": payload,
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
        "native_grasps": _native_grasps(controller),
        "pad_contacts": _pad_contacts(controller),
        "body_positions": body_positions,
        "body_target_distance": body_target_distances,
        "gripper_qpos": _gripper_qpos(env),
        "env_success": env_success,
    }


def _probe_action_response(adapter: Any, case: Any, spec: Any) -> list[dict[str, Any]]:
    """Measure xyz action response independently for both Transport arms."""
    result: list[dict[str, Any]] = []
    for arm_offset, arm_name in ((0, "arm0"), (7, "arm1")):
        for axis, axis_name in enumerate(("x", "y", "z")):
            state = adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
            before0 = _slice(state, spec, "robot0_eef_pos")
            before1 = _slice(state, spec, "robot1_eef_pos")
            action = np.zeros(adapter.action_dim, dtype=np.float64)
            action[arm_offset + axis] = 0.25
            action[arm_offset + 6] = -1.0
            after, _done, success, _info = adapter.step(action)
            after0 = _slice(after, spec, "robot0_eef_pos")
            after1 = _slice(after, spec, "robot1_eef_pos")
            result.append(
                {
                    "arm": arm_name,
                    "axis": axis_name,
                    "action": action,
                    "eef0_delta": None if before0 is None or after0 is None else after0 - before0,
                    "eef1_delta": None if before1 is None or after1 is None else after1 - before1,
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

    for t in range(max_steps):
        phase_before = controller.phase_name
        before = _snapshot(controller, state, adapter.env)
        action = np.asarray(controller.act(state, t), dtype=np.float64)
        command = {
            "arm0_xyz": action[:3].copy(),
            "arm0_gripper": float(action[6]),
            "arm1_xyz": action[7:10].copy(),
            "arm1_gripper": float(action[13]),
        }
        phase_after_action = controller.phase_name
        next_state, _done, success, info = adapter.step(action)
        after = _snapshot(controller, next_state, adapter.env)
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
            print(
                f"case={case.index:02d} t={t:03d} "
                f"phase={phase_before}->{controller.phase_name} "
                f"d0={distances.get('arm0', float('nan')):.4f} "
                f"d1={distances.get('arm1', float('nan')):.4f} "
                f"g={after['native_grasps']} "
                f"p={after.get('pad_contacts')} "
                f"success={bool(success)}"
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
            f"cases: {[case.index for case in selected]}; horizon: {adapter.horizon}"
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
