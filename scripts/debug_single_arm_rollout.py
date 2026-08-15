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
    parser.add_argument("--case-index", type=int, default=None)
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
            selected = [case for case in bank.cases if case.index == args.case_index]
            if not selected:
                raise ValueError(
                    f"case index {args.case_index} is not present in bank "
                    f"{bank.bank_id} (0..{len(bank.cases) - 1})"
                )
        else:
            selected = bank.cases[: args.cases]
        print(f"Task: {bank.task}; bank: {bank.bank_id}; cases: {[c.index for c in selected]}")
        controller_config = (
            None
            if args.descend_z_offset is None
            else ScriptedControllerConfig(descend_z_offset=args.descend_z_offset)
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
        "traces": traces,
    }
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Complete diagnostic trace written to {path}")


if __name__ == "__main__":
    main()
