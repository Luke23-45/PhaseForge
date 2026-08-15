"""Trace the state-only Lift scripted controller in the pinned robosuite env.

This is a diagnostic tool only. It does not modify the frozen reset bank, raw
data, checkpoints, or evaluation summaries. It is intended to answer one
question when the scripted-controller gate fails: is the controller reaching
the cube but failing to grasp it, or is the action/state mapping wrong?

Examples::

    uv run python scripts/debug_lift_rollout.py --cases 2 data=lift eval=rollout
    uv run python scripts/debug_lift_rollout.py --case-index 0 --log-every 1 data=lift eval=rollout

The script writes a complete JSON trace under ``outputs/_gates`` and prints a
compact per-step trace to stdout. It never downloads data and never regenerates
an existing frozen reset bank.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
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


def _controller_metadata(env: Any) -> dict[str, Any]:
    """Report the installed arm-controller contract without assuming a class."""
    result: dict[str, Any] = {}
    try:
        robots = getattr(env, "robots", [])
        for robot_index, robot in enumerate(robots):
            controllers = getattr(robot, "part_controllers", {})
            for arm_name, controller in controllers.items():
                result[f"robot{robot_index}.{arm_name}"] = {
                    name: _jsonable(getattr(controller, name, None))
                    for name in (
                        "__class__",
                        "input_type",
                        "input_ref_frame",
                        "control_delta",
                        "input_min",
                        "input_max",
                        "output_min",
                        "output_max",
                    )
                }
                result[f"robot{robot_index}.{arm_name}"]["__class__"] = type(
                    controller
                ).__name__
    except Exception as exc:  # noqa: BLE001 - diagnostics must continue
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _real_sim_metrics(env: Any) -> dict[str, Any]:
    """Read simulator-native object/grasp signals when this task exposes them."""
    metrics: dict[str, Any] = {}
    try:
        body_id = getattr(env, "cube_body_id")
        metrics["cube_body_pos"] = np.asarray(
            env.sim.data.body_xpos[int(body_id)], dtype=np.float64
        ).copy()
    except Exception:
        pass
    try:
        robot = env.robots[0]
        metrics["gripper_qpos_native"] = np.asarray(
            env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes], dtype=np.float64
        ).copy()
    except Exception:
        pass
    try:
        metrics["env_success"] = bool(env._check_success())
    except Exception:
        pass
    try:
        metrics["grasp_check"] = bool(
            env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cube)
        )
    except Exception:
        pass
    return metrics


def _eef_from_state(state: np.ndarray, spec) -> np.ndarray:
    start, stop = spec.index_of("robot0_eef_pos")
    return np.asarray(state[start:stop], dtype=np.float64).copy()


def _object_from_controller(controller: Any, state: np.ndarray) -> np.ndarray:
    return np.asarray(controller.object_pos(state), dtype=np.float64).copy()


def _probe_action_response(adapter: Any, case: Any, spec: Any) -> list[dict[str, Any]]:
    """Apply one small positive Cartesian action per axis from the same state."""
    probe: list[dict[str, Any]] = []
    for axis, name in enumerate(("x", "y", "z")):
        state = adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
        before = _eef_from_state(state, spec)
        action = np.zeros(adapter.action_dim, dtype=np.float64)
        action[axis] = 0.25
        action[6] = 1.0
        after, _done, success, _info = adapter.step(action)
        after_eef = _eef_from_state(after, spec)
        probe.append(
            {
                "axis": name,
                "action": action,
                "eef_before": before,
                "eef_after": after_eef,
                "eef_delta": after_eef - before,
                "success_after_step": bool(success),
            }
        )
    return probe


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
        "initial_state": state.copy(),
        "action_response_probe": (
            _probe_action_response(adapter, case, spec) if include_probe else []
        ),
        "steps": [],
        "success": False,
        "termination": "horizon",
    }
    # The probe leaves the environment at a different state; restore the
    # actual case before tracing the controller.
    state = adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)

    for t in range(max_steps):
        phase_before = controller.phase_name
        eef_before = _eef_from_state(state, spec)
        object_before = _object_from_controller(controller, state)
        action = np.asarray(controller.act(state, t), dtype=np.float64)
        target = getattr(controller, "_last_target", None)
        target_distance = getattr(controller, "_last_target_distance", None)
        next_state, _done, success, _info = adapter.step(action)
        eef_after = _eef_from_state(next_state, spec)
        object_after = _object_from_controller(controller, next_state)
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
            "object_z": float(object_after[2]),
            "action": action,
            "gripper_action": float(action[6]),
            "target": None if target is None else np.asarray(target).copy(),
            "target_distance_before_action": target_distance,
            "sim_metrics": _real_sim_metrics(adapter.env),
            "success": bool(success),
        }
        record["steps"].append(step)
        if t % max(1, log_every) == 0 or success or controller.phase_name == "STALLED":
            print(
                f"case={case.index:02d} t={t:03d} "
                f"phase={phase_before}->{controller.phase_name} "
                f"eef={np.round(eef_after, 5).tolist()} "
                f"obj={np.round(object_after, 5).tolist()} "
                f"a_xyz={np.round(action[:3], 3).tolist()} "
                f"grip={action[6]:.1f} "
                f"grasp={step['sim_metrics'].get('grasp_check', '?')} "
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
    return _jsonable(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=2, help="number of bank cases to trace")
    parser.add_argument("--case-index", type=int, default=None, help="trace one exact bank case")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-action-probe", action="store_true")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, for example data=lift eval=rollout",
    )
    args = parser.parse_args()
    if args.cases <= 0 or args.max_steps <= 0 or args.log_every <= 0:
        parser.error("--cases, --max-steps, and --log-every must be positive")

    overrides = args.overrides or ["data=lift", "eval=rollout"]
    with initialize_config_module(version_base="1.3", config_module="phaseforge.config"):
        cfg: DictConfig = compose(config_name="main", overrides=overrides)

    meta = resolve_pinned_metadata(cfg)
    spec = state_spec_from_config(cfg)
    adapter = _adapter_from_config(cfg, meta)
    try:
        bank = load_or_generate_bank(cfg, meta)
        controller_cls = TaskSpec.from_protocol(bank.task).get_controller_class()
        controller_metadata = _controller_metadata(adapter.env)
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
        print("Controller metadata:", json.dumps(_controller_metadata(adapter.env), indent=2))
        traces = [
            _trace_case(
                adapter,
                case,
                spec,
                controller_cls,
                max_steps=args.max_steps,
                log_every=args.log_every,
                include_probe=not args.no_action_probe,
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
        "data_root": str(Path(str(cfg.data.source.dir)).parent.parent.parent),
        "config_overrides": overrides,
        "controller_metadata": controller_metadata,
        "traces": traces,
    }
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Complete diagnostic trace written to {path}")


if __name__ == "__main__":
    main()
