"""Trace a single-arm state-only scripted rollout.

This diagnostic covers Can, Square, and ToolHang. It is deliberately separate
from the rollout gate and does not modify the frozen reset bank, raw data,
checkpoints, or evaluation summaries. It records the state-derived object and
target positions, both action and end-effector response, phase transitions,
the watchdog state, and the simulator's own success predicate.

Examples::

    uv run python scripts/debug_single_arm_rollout.py \
        --case-index 0 --max-steps 500 --log-every 5 \
        data=can eval=rollout

    uv run python scripts/debug_single_arm_rollout.py \
        --case-index 0 --max-steps 500 --log-every 5 \
        data=square eval=rollout

No images are requested or consumed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
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
from phaseforge.evaluations.rollout.scripted_controller import ScriptedControllerConfig
from phaseforge.utils.config import output_base_dir


def _jsonable(value: Any) -> Any:
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


def _state_slice(state: np.ndarray, spec: Any, key: str) -> np.ndarray | None:
    try:
        start, stop = spec.index_of(key)
    except (KeyError, ValueError):
        return None
    return np.asarray(state[start:stop], dtype=np.float64).copy()


def _native_metrics(env: Any, controller: Any, state: np.ndarray) -> dict[str, Any]:
    """Collect non-image simulator facts and preserve extraction errors."""
    metrics: dict[str, Any] = {}
    try:
        metrics["env_success"] = bool(env._check_success())
    except Exception as exc:  # noqa: BLE001 - diagnostics retain the cause
        metrics["env_success_error"] = f"{type(exc).__name__}: {exc}"
    try:
        metrics["controller_native_grasp_status"] = controller._native_grasp_status()
    except Exception as exc:  # noqa: BLE001 - diagnostics retain the cause
        metrics["controller_native_grasp_error"] = f"{type(exc).__name__}: {exc}"
    try:
        robot = env.robots[0]
        arms = getattr(robot, "arms", ())
        arm = arms[0] if arms else "right"
        indexes = getattr(robot, "_ref_gripper_joint_pos_indexes")
        if isinstance(indexes, dict):
            indexes = indexes[arm]
        metrics["gripper_qpos_native"] = np.asarray(
            env.sim.data.qpos[indexes], dtype=np.float64
        ).copy()
        metrics["gripper_arm"] = arm
    except Exception as exc:  # noqa: BLE001 - diagnostics retain the cause
        metrics["gripper_qpos_native_error"] = f"{type(exc).__name__}: {exc}"

    # NutAssembly-specific facts make Square failures distinguishable without
    # rendering images: active nut pose, peg pose, native on_peg result,
    # robosuite's cached objects_on_pegs flag, and every term used by the
    # controller's release guard.
    try:
        nuts = getattr(env, "nuts", None)
        nut_id = getattr(env, "nut_id", None)
        body_ids = getattr(env, "obj_body_id", {})
        if nuts is not None and nut_id is not None:
            nut = nuts[int(nut_id)]
            nut_pos = np.asarray(
                env.sim.data.body_xpos[body_ids[nut.name]], dtype=np.float64
            ).copy()
            metrics["nut_id_native"] = int(nut_id)
            metrics["nut_name_native"] = str(nut.name)
            metrics["nut_position_native"] = nut_pos
            metrics["eef_to_nut_distance_native"] = float(
                np.linalg.norm(controller.eef_pos(state) - nut_pos)
            )
            if hasattr(controller, "_square_yaw_error"):
                metrics["square_controller_yaw_error"] = float(
                    controller._square_yaw_error
                )
                try:
                    eef_quat_start, eef_quat_end = controller.state_spec.index_of(
                        "robot0_eef_quat"
                    )
                    object_start, object_end = controller.state_spec.index_of("object")
                    eef_quat = state[eef_quat_start:eef_quat_end]
                    object_vec = state[object_start:object_end]
                    if object_vec.shape[0] >= 14:
                        metrics["square_eef_yaw"] = float(
                            controller._yaw_from_xyzw(eef_quat)
                        )
                        metrics["square_object_yaw"] = float(
                            controller._yaw_from_xyzw(object_vec[10:14])
                        )
                except (AttributeError, TypeError, ValueError):
                    pass
            site_ids = getattr(env, "object_site_ids", None)
            if site_ids is not None:
                handle_pos = np.asarray(
                    env.sim.data.site_xpos[int(site_ids[int(nut_id)])],
                    dtype=np.float64,
                ).copy()
                metrics["nut_handle_position_native"] = handle_pos
                metrics["eef_to_nut_handle_distance_native"] = float(
                    np.linalg.norm(controller.eef_pos(state) - handle_pos)
                )
            peg_id = int(getattr(env, "peg1_body_id"))
            peg_pos = np.asarray(env.sim.data.body_xpos[peg_id], dtype=np.float64).copy()
            metrics["peg1_position_native"] = peg_pos
            metrics["on_peg_native"] = bool(env.on_peg(nut_pos, int(nut_id)))
            metrics["objects_on_pegs_native"] = np.asarray(
                getattr(env, "objects_on_pegs", []), dtype=np.float64
            ).copy()
            eef_pos = controller.eef_pos(state)
            metrics["square_xy_error_to_peg"] = float(
                np.linalg.norm(nut_pos[:2] - peg_pos[:2])
            )
            metrics["square_nut_below_peg"] = bool(
                float(nut_pos[2]) < float(peg_pos[2]) + 0.015
            )
            metrics["square_eef_clear_of_nut"] = bool(
                float(np.linalg.norm(eef_pos - nut_pos)) > 0.045
            )
            metrics["square_release_ready"] = bool(
                controller._placement_release_ready(state)
            )
            metrics["square_eef_to_peg_distance"] = float(
                np.linalg.norm(eef_pos - peg_pos)
            )
    except Exception as exc:  # noqa: BLE001 - diagnostics retain the cause
        metrics["nut_geometry_native_error"] = f"{type(exc).__name__}: {exc}"

    # ToolHang-specific facts expose the complete geometry used by
    # robosuite's success predicate: tool body, tool-hole center, hook line,
    # frame assembly, and whether the released tool is still touching the
    # gripper. This keeps a failed gate diagnosable from one trace.
    try:
        site_ids = getattr(env, "obj_site_id", {})
        body_ids = getattr(env, "obj_body_id", {})
        hole_pos: np.ndarray | None = None
        if "tool" in body_ids and "tool_hole1_center" in site_ids:
            tool_pos = np.asarray(
                env.sim.data.body_xpos[int(body_ids["tool"])], dtype=np.float64
            ).copy()
            hole_pos = np.asarray(
                env.sim.data.site_xpos[int(site_ids["tool_hole1_center"])],
                dtype=np.float64,
            ).copy()
            metrics["tool_position_native"] = tool_pos
            metrics["tool_hole_position_native"] = hole_pos
            metrics["eef_to_tool_distance_native"] = float(
                np.linalg.norm(controller.eef_pos(state) - tool_pos)
            )
            metrics["eef_to_tool_hole_distance_native"] = float(
                np.linalg.norm(controller.eef_pos(state) - hole_pos)
            )
        if "frame_hang_site" in site_ids:
            hook_pos = np.asarray(
                env.sim.data.site_xpos[int(site_ids["frame_hang_site"])],
                dtype=np.float64,
            ).copy()
            metrics["frame_hang_site_position_native"] = hook_pos
            if hole_pos is not None:
                metrics["tool_hole_to_hook_distance_native"] = float(
                    np.linalg.norm(hole_pos - hook_pos)
                )
        if hasattr(env, "_check_frame_assembled"):
            metrics["frame_assembled_native"] = bool(env._check_frame_assembled())
        if hasattr(env, "_check_tool_on_frame"):
            metrics["tool_on_frame_native"] = bool(env._check_tool_on_frame())
    except Exception as exc:  # noqa: BLE001 - diagnostics retain the cause
        metrics["tool_geometry_native_error"] = f"{type(exc).__name__}: {exc}"
    metrics["object_state"] = _jsonable(state)
    return metrics


def _probe_action_response(adapter: Any, case: Any, spec: Any) -> list[dict[str, Any]]:
    """Measure positive xyz action response from the exact reset state."""
    result: list[dict[str, Any]] = []
    for axis, axis_name in enumerate(("x", "y", "z")):
        state = adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
        before = _state_slice(state, spec, "robot0_eef_pos")
        if before is None:
            raise ValueError("state schema has no robot0_eef_pos field")
        action = np.zeros(adapter.action_dim, dtype=np.float64)
        action[axis] = 0.25
        action[6] = -1.0
        after, _done, success, _info = adapter.step(action)
        after_eef = _state_slice(after, spec, "robot0_eef_pos")
        assert after_eef is not None
        result.append(
            {
                "axis": axis_name,
                "action": action,
                "eef_before": before,
                "eef_after": after_eef,
                "eef_delta": after_eef - before,
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
    controller_config: ScriptedControllerConfig | None = None,
) -> dict[str, Any]:
    controller = controller_cls(
        spec,
        env=adapter.env,
        config=controller_config,
    )
    state = adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
    eef = _state_slice(state, spec, "robot0_eef_pos")
    if eef is None:
        raise ValueError("state schema has no robot0_eef_pos field")
    record: dict[str, Any] = {
        "case_index": int(case.index),
        "initial_state": state.copy(),
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
        eef_before = _state_slice(state, spec, "robot0_eef_pos")
        object_before = controller.object_pos(state)
        target = controller._placement_target()
        action = np.asarray(controller.act(state, t), dtype=np.float64)
        next_state, _done, success, _info = adapter.step(action)
        eef_after = _state_slice(next_state, spec, "robot0_eef_pos")
        object_after = controller.object_pos(next_state)
        target_after = controller._placement_target()
        target_distance = (
            None
            if target_after is None
            else float(np.linalg.norm(np.asarray(target_after) - eef_after))
        )
        sim_metrics = _native_metrics(adapter.env, controller, next_state)
        step = {
            "t": t,
            "phase_before": phase_before,
            "phase_after": controller.phase_name,
            "stalled_from_phase": controller.stalled_from_phase,
            "eef_before": eef_before,
            "eef_after": eef_after,
            "eef_delta": eef_after - eef_before,
            "object_before": object_before,
            "object_after": object_after,
            "target_before": target,
            "target_after": target_after,
            "target_distance_after": target_distance,
            "action": action,
            "gripper_action": float(action[6]),
            "release_commanded": bool(
                phase_before in {"PLACE", "RETRACT"} and action[6] < 0.0
            ),
            "sim_metrics": sim_metrics,
            "success": bool(success),
        }
        record["steps"].append(step)
        if (
            t % max(1, log_every) == 0
            or success
            or phase_before != controller.phase_name
            or controller.phase_name == "STALLED"
        ):
            print(
                f"case={case.index:02d} t={t:03d} "
                f"phase={phase_before}->{controller.phase_name} "
                f"eef={np.round(eef_after, 5).tolist()} "
                f"obj={np.round(object_after, 5).tolist()} "
                f"target={None if target_after is None else np.round(target_after, 5).tolist()} "
                f"target_dist={target_distance} "
                f"a_xyz={np.round(action[:3], 3).tolist()} "
                f"grip={action[6]:+.1f} "
                f"grasp={sim_metrics.get('controller_native_grasp_status')} "
                f"env_success={sim_metrics.get('env_success')}"
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
    return _jsonable(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1)
    parser.add_argument(
        "--case-index",
        type=int,
        action="append",
        help="trace an exact bank case; repeat for multiple cases",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--no-action-probe", action="store_true")
    parser.add_argument(
        "--descend-z-offset",
        type=float,
        default=None,
        help="optional diagnostic-only grasp offset override in metres",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides")
    args = parser.parse_args()
    if args.cases <= 0 or args.max_steps <= 0 or args.log_every <= 0:
        parser.error("--cases, --max-steps, and --log-every must be positive")
    if args.case_index is not None and any(index < 0 for index in args.case_index):
        parser.error("--case-index values must be non-negative")
    if args.descend_z_offset is not None and (
        not np.isfinite(args.descend_z_offset) or args.descend_z_offset < 0.0
    ):
        parser.error("--descend-z-offset must be a finite non-negative number")

    overrides = args.overrides or ["data=can", "eval=rollout"]
    with initialize_config_module(version_base="1.3", config_module="phaseforge.config"):
        cfg: DictConfig = compose(config_name="main", overrides=overrides)

    meta = resolve_pinned_metadata(cfg)
    spec = state_spec_from_config(cfg)
    adapter = _adapter_from_config(cfg, meta)
    try:
        bank = load_or_generate_bank(cfg, meta)
        if bank.task not in {"Can", "Square", "ToolHang"}:
            raise ValueError(
                "This diagnostic is for single-arm placement tasks "
                f"(Can, Square, ToolHang), got {bank.task!r}"
            )
        controller_cls = TaskSpec.from_protocol(bank.task).get_controller_class()
        if args.case_index is not None:
            invalid = [index for index in args.case_index if index >= bank.num_cases]
            if invalid:
                raise ValueError(
                    f"case index {invalid} outside bank range "
                    f"0..{bank.num_cases - 1}"
                )
            selected = [bank.cases[index] for index in args.case_index]
        else:
            selected = bank.cases[: args.cases]
        print(f"Task: {bank.task}; bank: {bank.bank_id}; cases: {[c.index for c in selected]}")
        # Preserve task-specific defaults (notably Square's short grasp hold)
        # when applying a diagnostic-only offset override.
        default_controller = controller_cls(spec, env=adapter.env)
        controller_config = (
            default_controller.config
            if args.descend_z_offset is None
            else replace(
                default_controller.config,
                descend_z_offset=args.descend_z_offset,
            )
        )
        effective_controller = controller_cls(
            spec,
            env=adapter.env,
            config=controller_config,
        )
        effective_descend_z_offset = float(
            effective_controller.config.descend_z_offset
        )
        traces = [
            _trace_case(
                adapter,
                case,
                spec,
                controller_cls,
                max_steps=args.max_steps,
                log_every=args.log_every,
                include_probe=not args.no_action_probe,
                controller_config=controller_config,
            )
            for case in selected
        ]
    finally:
        adapter.close()

    output_dir = output_base_dir(cfg) / "_gates"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"debug_{bank.task.lower()}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    payload = {
        "task": bank.task,
        "bank_id": bank.bank_id,
        "config_overrides": overrides,
        "diagnostic_descend_z_offset": args.descend_z_offset,
        "effective_descend_z_offset": effective_descend_z_offset,
        "traces": traces,
    }
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Complete diagnostic trace written to {path}")


if __name__ == "__main__":
    main()
