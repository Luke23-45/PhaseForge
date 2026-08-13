"""End-to-end simulation of the PhaseForge data pipeline.

Goal: WITHOUT the real 66GB LIBERO download, verify that the pipeline
actually produces correct processed data. We synthesize tiny HDF5 files
that match the REAL LIBERO schema, then run the production FSM code
against them and report every failure.

This file is a throwaway verification harness — it lives outside the
package and exercises the real phaseforge.* code paths.

Schema used: "flattened" naming (obs/joint_states, obs/ee_pos,
obs/gripper_states, demo/robot_states). This is the schema found
on the HuggingFace mirror (yifengzhu-hf/LIBERO-datasets) per
multiple independent sources. The VisionStripper maps these
flattened keys to the config's robot0_* key names explicitly.
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

# Repo root on disk
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Synthetic object-state layout (mirrors the census-built index format)
# ---------------------------------------------------------------------------

#: Number of synthetic objects per task (all free joints).
SYNTHETIC_OBJECTS = 3
#: qpos size: robot (7 arm + 2 gripper) + 3 free objects x 7.
SYNTHETIC_NQ = 9 + 7 * SYNTHETIC_OBJECTS
#: qvel size: robot (7 + 2) + 3 free objects x 6.
SYNTHETIC_NV = 9 + 6 * SYNTHETIC_OBJECTS

#: Object names per synthetic task (3 tasks, mirroring step1_synthesize).
SYNTHETIC_OBJECTS_BY_TASK = {
    "KITCHEN_SCENE1_open_drawer_demo": ["drawer_0", "bowl_0", "book_0"],
    "LIVING_ROOM_SCENE2_put_bowl_demo": ["bowl_0", "cup_0", "plate_0"],
    "STUDY_SCENE1_pick_book_demo": ["book_0", "laptop_0", "pen_0"],
}


def synthetic_states(T: int, rng: np.random.Generator) -> np.ndarray:
    """Build a ``(T, nq+nv)`` flattened sim state for the synthetic tasks.

    Layout matches the real LIBERO convention: ``[qpos(nq), qvel(nv)]``.
    Object free-joint blocks are normalized quaternions so the decode math
    is exercised on valid inputs.
    """
    qpos = np.empty((T, SYNTHETIC_NQ), dtype=np.float32)
    qpos[:, :7] = rng.normal(0, 0.5, (T, 7))    # arm joints
    qpos[:, 7:9] = 0.05                          # gripper (open-ish)
    for k in range(SYNTHETIC_OBJECTS):
        base = 9 + 7 * k
        qpos[:, base : base + 3] = (
            np.cumsum(rng.normal(0, 0.01, (T, 3)), axis=0) + 1.0
        )
        qpos[:, base + 3] = 1.0
        qpos[:, base + 4 : base + 7] = rng.normal(0, 0.05, (T, 3))
        norm = np.linalg.norm(qpos[:, base + 3 : base + 7], axis=-1, keepdims=True)
        qpos[:, base + 3 : base + 7] /= norm
    qvel = rng.normal(0, 0.01, (T, SYNTHETIC_NV)).astype(np.float32)
    return np.concatenate([qpos, qvel], axis=-1).astype(np.float32)


def write_synthetic_object_index(
    raw_libero_dir: Path,
    object_names_by_task: dict[str, list[str]],
    k_slots: int = 16,
) -> Path:
    """Write an ``object_index.json`` for synthetic tasks (all free joints).

    Mirrors the census format consumed by ``ObjectIndex.load``. The index
    lives next to the raw suite dirs, as the real census output does.
    """
    tasks = {
        name: {
            "nq": SYNTHETIC_NQ,
            "objects": [
                {
                    "name": obj,
                    "joint_type": "free",
                    "qpos_start": 9 + 7 * k,
                    "qpos_len": 7,
                }
                for k, obj in enumerate(names)
            ],
        }
        for name, names in object_names_by_task.items()
    }
    idx_path = raw_libero_dir / "object_index.json"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(
        json.dumps(
            {
                "version": 1,
                "k_slots": k_slots,
                "dim_per_object": 7,
                "include_mask": True,
                "tasks": tasks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return idx_path


# ---------------------------------------------------------------------------
# Step 1: synthesize LIBERO-shaped HDF5 files
# ---------------------------------------------------------------------------

def make_synthetic_libero_file(
    path: Path, n_demos: int = 3, T: int = 60, seed: int = 0
):
    """Write one HDF5 file matching the real LIBERO "flattened" schema.

    Structure::
        /data/demo_{i}/
            obs/
                joint_states       (T, 7)   float32
                ee_pos             (T, 3)   float32
                gripper_states     (T, 2)   float32
                agentview_rgb      (T, H, W, 3) uint8   (vision — stripped)
                eye_in_hand_rgb    (T, H, W, 3) uint8   (vision — stripped)
            robot_states           (T, 9)   float32  [gripper(2), eef_pos(3), eef_quat(4)]
            states                 (T, 57)  float32  [qpos(30), qvel(27)] — object decode
            actions                (T, 7)   float32

    We craft a trajectory with a deliberate grasp-release cycle so the
    rule-based phase labeler has real transitions to detect:
      - gripper open  for first third  (APPROACH)
      - gripper closes at 1/3          (GRASP)
      - gripper closed mid             (TRANSPORT)
      - gripper opens  at 2/3          (PLACE)
      - gripper open  after            (RETRACT)
    """
    rng = np.random.default_rng(seed)
    H, W = 128, 128  # typical LIBERO image resolution
    with h5py.File(path, "w") as f:
        data_grp = f.create_group("data")
        for d in range(n_demos):
            grp = data_grp.create_group(f"demo_{d}")
            obs = grp.create_group("obs")

            # Joint states (7-DoF)
            joint_states = rng.normal(0, 0.5, (T, 7)).astype(np.float32)
            obs["joint_states"] = joint_states

            # EE position — make it actually move so velocity > threshold
            eef_pos = np.cumsum(
                rng.normal(0, 0.02, (T, 3)), axis=0
            ).astype(np.float32)
            obs["ee_pos"] = eef_pos

            # Gripper qpos — the critical phase signal
            gripper = np.ones((T, 2), np.float32) * 0.06  # "open"
            t1, t2 = T // 3, (2 * T) // 3
            gripper[t1:t2] = 0.005  # "closed" (below 0.02 threshold)
            obs["gripper_states"] = gripper

            # Vision keys — required by the flattened schema; the stripper
            # must drop them without ever loading them into RAM.
            obs["agentview_rgb"] = np.zeros((T, H, W, 3), dtype=np.uint8)
            obs["eye_in_hand_rgb"] = np.zeros((T, H, W, 3), dtype=np.uint8)

            # robot_states at demo root (9-dim: gripper + eef_pos + eef_quat)
            # quat = identity for simplicity
            eef_quat = np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], np.float32), (T, 1)
            )
            robot_states = np.concatenate(
                [gripper, eef_pos, eef_quat], axis=-1
            ).astype(np.float32)
            grp["robot_states"] = robot_states

            # Flattened sim state (P-Stage 1 object-state decode source)
            grp["states"] = synthetic_states(T, rng)

            # Actions: 7-dim
            grp["actions"] = rng.normal(0, 0.05, (T, 7)).astype(np.float32)


def step1_synthesize(sim_root: Path) -> Path:
    print("\n========== STEP 1: synthesize LIBERO-shaped HDF5 ==========")
    raw_suite = sim_root / "raw" / "libero" / "libero_90"
    raw_suite.mkdir(parents=True, exist_ok=True)
    task_names = list(SYNTHETIC_OBJECTS_BY_TASK)
    for i, task in enumerate(task_names):
        p = raw_suite / f"{task}.hdf5"
        make_synthetic_libero_file(p, n_demos=3, T=60, seed=i)
        print(f"  wrote {p.name}  ({3} demos x 60 steps)")
    idx_path = write_synthetic_object_index(
        raw_suite.parent, SYNTHETIC_OBJECTS_BY_TASK
    )
    print(f"  wrote object index {idx_path.name} (k_slots=16)")
    print(f"  -> {raw_suite}")
    return raw_suite


# ---------------------------------------------------------------------------
# Step 2: build a Hydra-style config that the FSM expects
# ---------------------------------------------------------------------------

def step2_build_config(sim_root: Path, raw_suite: Path) -> OmegaConf:
    print("\n========== STEP 2: build config ==========")
    cfg = OmegaConf.create({
        "project": {
            "output_dir": str(sim_root / "outputs"),
        },
        "data": {
            "cache_root": "cache",
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "sequence_length": 1,
            "stride": 1,
            "state_dim": 151,
            "state_keys": [
                {"key": "robot0_joint_pos", "dim": 7},
                {"key": "robot0_joint_vel", "dim": 7},
                {"key": "robot0_eef_pos", "dim": 3},
                {"key": "robot0_eef_quat", "dim": 4},
                {"key": "robot0_gripper_qpos", "dim": 2},
            ],
            "object_state": {
                "enabled": True,
                "k_slots": 16,
                "dim_per_object": 7,
                "include_mask": True,
                "index_path": str(raw_suite.parent / "object_index.json"),
            },
            "libero": {
                "local": {"raw_dir": "raw_data/libero90"},
                "split": {
                    "train_ratio": 0.9, "val_ratio": 0.1, "test_ratio": 0.0,
                    "seed": 42,
                },
                "phase_labeler": {
                    "_target_": "phaseforge.data.libero.phase_labeler.RuleBasedPhaseLabeler",
                    "num_phases": 6,
                    "gripper_closed_threshold": 0.02,
                    "gripper_open_threshold": 0.04,
                    "eef_velocity_threshold": 0.01,
                    "min_phase_duration": 5,
                    "median_filter_size": 7,
                },
            },
        },
    })
    print("  config built.")
    return cfg


# ---------------------------------------------------------------------------
# Step 3: run the REAL FSM and capture every failure
# ---------------------------------------------------------------------------

def step3_run_fsm(cfg, raw_suite: Path):
    """Run the FSM, bypassing validate_source by pointing at synthetic files.

    The current FSM validates file count (expects 90 for libero_90), but
    simulation only has 3 files. We monkeypatch `_validate_source` to
    point at the synthetic suite and skip the count check.
    """
    print("\n========== STEP 3: run FSM (download bypassed) ==========")
    from phaseforge.data.ingestion.state_machine import DataPipelineStateMachine
    from phaseforge.data.ingestion.states import PipelineState

    pipeline = DataPipelineStateMachine(cfg)
    print(f"  initial state: {pipeline._state.name}")
    print(f"  config_hash: {pipeline.config_hash}")

    # Bypass validate_source: use synthetic suite, skip file-count check.
    # Also need to build the task index since validate_source would do it.
    from phaseforge.data.libero.task_index import build_task_index

    task_index = build_task_index(raw_suite)

    def _bypass_validate(self):
        self._raw_dir = raw_suite
        self._task_index = task_index
        self._state = PipelineState.INGEST_AND_STRIP

    pipeline._validate_source = _bypass_validate.__get__(pipeline)
    print("  [BYPASS] patched _validate_source to use synthetic HDF5")

    try:
        loaders = pipeline.run()
        print("  run() returned without error.")
        for name, loader in loaders.items():
            print(f"    {name}: {type(loader).__name__ if loader else 'None'}")
        return loaders
    except Exception as e:
        print(f"  !!! run() FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Step 4: if loaders exist, validate the actual tensor shapes/values
# ---------------------------------------------------------------------------

def step4_validate(loaders):
    print("\n========== STEP 4: validate output tensors ==========")
    if loaders is None:
        print("  skipped (FSM failed)")
        return
    train = loaders.get("train")
    if train is None:
        print("  no train loader")
        return
    batch = next(iter(train))
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"    {k}: {v}")


def main():
    sim_root = REPO / "_simulation"
    if sim_root.exists():
        shutil.rmtree(sim_root)
    sim_root.mkdir()

    raw_suite = step1_synthesize(sim_root)
    cfg = step2_build_config(sim_root, raw_suite)
    loaders = step3_run_fsm(cfg, raw_suite)
    step4_validate(loaders)

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()
