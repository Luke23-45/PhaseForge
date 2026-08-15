"""Diagnostic script to run on the Colab/remote machine with robosuite.

Run with:
    uv run python scripts/diagnose_gate4.py

This script:
1. Creates the adapter exactly as the gates do
2. Runs ONE episode with the scripted Lift controller
3. Prints every phase transition and key measurements
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra import compose, initialize

from phaseforge.evaluations.envs.env_metadata import PinnedEnvMetadata
from phaseforge.evaluations.envs.robosuite_adapter import RobosuiteStateAdapter, StateSpec
from phaseforge.evaluations.rollout.reset_bank import load_reset_bank
from phaseforge.evaluations.rollout.scripted_controller import (
    ScriptedControllerConfig,
    ScriptedLiftController,
)


def main() -> None:
    with initialize(version_base="1.3", config_path="../phaseforge/config"):
        cfg = compose(config_name="main", overrides=["data=lift", "eval=rollout"])

    # Build state spec from config
    keys = tuple(item.key for item in cfg.data.state_keys)
    dims = tuple(item.dim for item in cfg.data.state_keys)
    state_spec = StateSpec(keys=keys, dims=dims)

    # Load pinned env metadata from the HDF5 or dev fallback
    source_dir = Path(cfg.data.source.dir)
    hdf5_files = list(source_dir.glob("*.hdf5")) + list(source_dir.glob("**/*.hdf5"))
    if hdf5_files:
        meta = PinnedEnvMetadata.from_hdf5(hdf5_files[0])
    else:
        from phaseforge.evaluations.envs.task_registry import dev_fallback_metadata

        meta = dev_fallback_metadata("Lift")
    print(f"Env: {meta.env_name}, horizon: {meta.horizon}")
    print(f"State spec: keys={keys}, dims={dims}, total={sum(dims)}")

    # Build adapter
    adapter = RobosuiteStateAdapter(
        meta,
        state_spec,
        action_dim=cfg.data.action_dim,
    )

    # Load the reset bank
    from phaseforge.utils.config import output_base_dir

    base = output_base_dir(cfg)
    bank_dir = base / "_reset_banks"
    banks = list(bank_dir.glob("*.json"))
    if not banks:
        print("No reset bank found — generating one from the first demo")
        # Just use a fresh reset from the env
        adapter.env.reset()
        init_state = np.asarray(adapter.env.sim.get_state().flatten(), dtype=np.float64)
        state = adapter.reset_to(init_state)
    else:
        bank = load_reset_bank(banks[0])
        case = bank.cases[0]
        state = adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)

    print("\n--- Initial state ---")
    eef_start, eef_end = state_spec.index_of("robot0_eef_pos")
    obj_start, obj_end = state_spec.index_of("object")
    grip_start, grip_end = state_spec.index_of("robot0_gripper_qpos")

    eef = state[eef_start:eef_end]
    obj_full = state[obj_start:obj_end]
    grip = state[grip_start:grip_end]
    print(f"  eef_pos:      {eef}")
    print(f"  object[0:3]:  {obj_full[:3]}")
    print(f"  object[0:10]: {obj_full}")
    print(f"  gripper_qpos: {grip}")
    print(f"  eef z - obj z = {eef[2] - obj_full[2]:.4f}")

    # Also read the raw sim body positions
    cube_body_id = adapter.env.cube_body_id
    cube_pos = adapter.env.sim.data.body_xpos[cube_body_id].copy()
    print(f"  sim cube body_xpos: {cube_pos}")
    print(f"  table offset: {adapter.env.model.mujoco_arena.table_offset}")

    # Run the controller
    config = ScriptedControllerConfig()
    controller = ScriptedLiftController(state_spec, env=adapter.env, config=config)
    print(
        f"\nController config: descend_z_offset={config.descend_z_offset}, "
        f"approach_z_offset={config.approach_z_offset}, "
        f"position_tolerance={config.position_tolerance}"
    )

    last_phase = None
    for t in range(500):
        action = controller.act(state, t)
        phase = controller.phase_name

        if phase != last_phase:
            eef = state[eef_start:eef_end]
            grip = state[grip_start:grip_end]
            cube_pos = adapter.env.sim.data.body_xpos[cube_body_id].copy()
            print(f"\n  [t={t:3d}] Phase: {last_phase} -> {phase}")
            print(f"    eef_pos:        {eef}")
            print(f"    object[0:3]:    {state[obj_start : obj_start + 3]}")
            print(f"    sim cube pos:   {cube_pos}")
            print(f"    gripper_qpos:   {grip}")
            print(f"    action[0:3]:    {action[:3]}")
            print(f"    action[6]:      {action[6]} ({'CLOSE' if action[6] < 0 else 'OPEN'})")
            print(f"    eef-cube dz:    {eef[2] - cube_pos[2]:.4f}")
            last_phase = phase

        state, _done, success, _info = adapter.step(action)
        obj_full = state[obj_start:obj_end]

        # Check robosuite success
        if success:
            cube_pos = adapter.env.sim.data.body_xpos[cube_body_id].copy()
            print(f"\n  [t={t:3d}] SUCCESS! cube_pos={cube_pos}")
            break

        # Print periodic status during LIFT
        if phase == "LIFT" and t % 50 == 0:
            eef = state[eef_start:eef_end]
            cube_pos = adapter.env.sim.data.body_xpos[cube_body_id].copy()
            grip = state[grip_start:grip_end]
            check = adapter.env._check_success()
            print(
                f"  [t={t:3d}] LIFT status: eef_z={eef[2]:.4f}, "
                f"cube_z={cube_pos[2]:.4f}, grip={grip}, "
                f"_check_success={check}"
            )
    else:
        eef = state[eef_start:eef_end]
        cube_pos = adapter.env.sim.data.body_xpos[cube_body_id].copy()
        check = adapter.env._check_success()
        print(
            f"\n  TIMEOUT at t=500. Final: eef_z={eef[2]:.4f}, "
            f"cube_z={cube_pos[2]:.4f}, _check_success={check}"
        )
        print(f"  Final phase: {controller.phase_name}")
        if controller.stalled_from_phase:
            print(f"  Stalled from: {controller.stalled_from_phase}")

    adapter.close()


if __name__ == "__main__":
    main()
