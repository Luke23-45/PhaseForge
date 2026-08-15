"""Inspect human demonstration grasp mechanics and test controller parameter sweep.

Usage:
    uv run python scripts/inspect_demo_grasps.py

This script:
1. Inspects raw HDF5 demonstration trajectories to extract exact human grasp poses:
   - eef_pos at the moment gripper closes
   - object_pos at the moment gripper closes
   - exact delta (eef_z - object_z) used in human demonstrations
2. Simulates the scripted controller on the real robosuite environment across
   a grid of descend_z_offset and position_tolerance to find 100% solve rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra import compose, initialize

from phaseforge.evaluations.envs.env_metadata import PinnedEnvMetadata
from phaseforge.evaluations.envs.robosuite_adapter import RobosuiteStateAdapter, StateSpec
from phaseforge.evaluations.rollout.scripted_controller import (
    ScriptedControllerConfig,
    ScriptedLiftController,
)


def inspect_hdf5_grasps(hdf5_path: Path) -> dict[str, float]:
    print("\n==================================================")
    print(f"1. INSPECTING HUMAN DEMONSTRATIONS: {hdf5_path.name}")
    print("==================================================")
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        demo_keys = sorted(data.keys())[:10]
        dz_at_grasp = []
        eef_z_at_grasp = []
        cube_z_at_grasp = []

        for key in demo_keys:
            demo = data[key]
            actions = demo["actions"][:]
            obs = demo["obs"]
            eef_pos = obs["robot0_eef_pos"][:]
            # Lift stores cube pos in 'object' [0:3] or 'cube_pos'
            if "object" in obs:
                cube_pos = obs["object"][:, :3]
            elif "cube_pos" in obs:
                cube_pos = obs["cube_pos"][:]
            else:
                continue

            # robosuite PandaGripper uses +1 for closed and -1 for open.
            closed_steps = np.where(actions[:, -1] > 0.5)[0]
            if len(closed_steps) == 0:
                continue
            first_close = closed_steps[0]
            eef_z = eef_pos[first_close, 2]
            cube_z = cube_pos[first_close, 2]
            dz = eef_z - cube_z
            dz_at_grasp.append(dz)
            eef_z_at_grasp.append(eef_z)
            cube_z_at_grasp.append(cube_z)
            eef_x, eef_y = eef_pos[first_close, 0], eef_pos[first_close, 1]
            cube_x, cube_y = cube_pos[first_close, 0], cube_pos[first_close, 1]
            print(
                f"  {key} (t={first_close:3d}): "
                f"eef=({eef_x:.3f}, {eef_y:.3f}, {eef_z:.4f})  "
                f"cube=({cube_x:.3f}, {cube_y:.3f}, {cube_z:.4f})  "
                f"dz={dz:+.4f} m"
            )

        mean_dz = float(np.mean(dz_at_grasp))
        median_dz = float(np.median(dz_at_grasp))
        print(f"\n---> Dataset grasp stats over {len(dz_at_grasp)} demos:")
        print(f"     Mean (eef_z - cube_z):   {mean_dz:+.4f} m ({mean_dz * 100:+.2f} cm)")
        print(f"     Median (eef_z - cube_z): {median_dz:+.4f} m ({median_dz * 100:+.2f} cm)")
        min_eef, max_eef = min(eef_z_at_grasp), max(eef_z_at_grasp)
        min_cube, max_cube = min(cube_z_at_grasp), max(cube_z_at_grasp)
        print(f"     Min/Max eef_z:           [{min_eef:.4f}, {max_eef:.4f}]")
        print(f"     Min/Max cube_z:          [{min_cube:.4f}, {max_cube:.4f}]")
        return {"mean_dz": mean_dz, "median_dz": median_dz}


def test_parameter_sweep(
    adapter: RobosuiteStateAdapter,
    bank,
    state_spec: StateSpec,
    recommended_dz: float,
) -> None:
    print("\n==================================================")
    print("2. TESTING SCRIPTED CONTROLLER PARAMETER SWEEP")
    print("==================================================")

    # Test candidate descend offsets around dataset recommendation
    candidates = [
        recommended_dz,
        recommended_dz - 0.005,
        recommended_dz + 0.005,
        0.015,
        0.020,
        0.025,
    ]
    # Deduplicate candidates while preserving float order
    candidates = sorted(list({round(c, 4) for c in candidates}))

    for dz_offset in candidates:
        for pos_tol in [0.03, 0.025, 0.02]:
            cfg = ScriptedControllerConfig(
                descend_z_offset=dz_offset,
                position_tolerance=pos_tol,
                grasp_hold_steps=20,
                lift_z=0.95,
            )
            successes = 0
            timeout_phases = {}
            # Test on first 10 cases from bank
            test_cases = bank.cases[:10]
            for case in test_cases:
                ctrl = ScriptedLiftController(state_spec, env=adapter.env, config=cfg)
                state = adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
                ok = False
                for t in range(adapter.horizon):
                    action = ctrl.act(state, t)
                    state, _done, success, _info = adapter.step(action)
                    if success:
                        ok = True
                        break
                if ok:
                    successes += 1
                else:
                    p = ctrl.phase_name
                    timeout_phases[p] = timeout_phases.get(p, 0) + 1

            rate = successes / len(test_cases)
            print(
                f"  descend_z_offset={dz_offset:+.4f} m, pos_tol={pos_tol:.3f} m  --> "
                f"Solved: {successes:2d}/{len(test_cases)} ({rate * 100:5.1f}%) "
                f"{'FAIL: ' + str(timeout_phases) if rate < 1.0 else '[PASS]'}"
            )
            if rate == 1.0:
                print(f"\n>>> OPTIMAL: descend_z_offset={dz_offset}, position_tolerance={pos_tol}")
                break
        if rate == 1.0:
            break


def main() -> None:
    with initialize(version_base="1.3", config_path="../phaseforge/config"):
        cfg = compose(config_name="main", overrides=["data=lift", "eval=rollout"])

    keys = tuple(item.key for item in cfg.data.state_keys)
    dims = tuple(item.dim for item in cfg.data.state_keys)
    state_spec = StateSpec(keys=keys, dims=dims)

    source_dir = Path(cfg.data.source.dir)
    hdf5_files = list(source_dir.glob("*.hdf5")) + list(source_dir.glob("**/*.hdf5"))
    if not hdf5_files:
        print(f"Error: No HDF5 found in {source_dir}")
        return

    hdf5_path = hdf5_files[0]
    stats = inspect_hdf5_grasps(hdf5_path)
    recommended_dz = stats.get("median_dz", 0.02)

    meta = PinnedEnvMetadata.from_hdf5(hdf5_path)
    adapter = RobosuiteStateAdapter(meta, state_spec, action_dim=cfg.data.action_dim)

    try:
        from phaseforge.evaluations.rollout.runner import load_or_generate_bank

        bank = load_or_generate_bank(cfg, meta)
    except Exception as exc:
        print(f"Bank fallback ({exc}) — sampling fresh resets from adapter...")
        from phaseforge.evaluations.rollout.reset_bank import ResetCase

        cases = []
        for i in range(10):
            adapter.env.reset()
            st = np.asarray(adapter.env.sim.get_state().flatten(), dtype=np.float32)
            cases.append(ResetCase(index=i, states=st))

        class _SimpleBank:
            def __init__(self, c: list[ResetCase]) -> None:
                self.cases = c

        bank = _SimpleBank(cases)

    test_parameter_sweep(adapter, bank, state_spec, recommended_dz)
    adapter.close()


if __name__ == "__main__":
    main()
