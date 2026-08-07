"""Generate a small synthetic LIBERO-90 dataset for CPU dry runs.

Writes exactly ``EXPECTED_FILE_COUNTS["libero_90"]`` (90) tiny HDF5 files in
the real LIBERO "flattened" schema, so the production FSM's official 90-file
integrity check passes and the *unmodified* CLI pipelines can be exercised
end to end — no GPU, no 66GB download, no libero sim.

Schema mirrors ``scripts/simulate_pipeline.py`` (HF-mirror naming), which the
``VisionStripper`` maps to the config's ``robot0_*`` key names:

    /data/demo_{i}/
        obs/joint_states       (T, 7)   float32
        obs/ee_pos             (T, 3)   float32   (cumsum so it moves)
        obs/gripper_states     (T, 2)   float32   (open -> closed -> open)
        obs/agentview_rgb      (T,128,128,3) uint8  (vision — stripped)
        obs/eye_in_hand_rgb    (T,128,128,3) uint8  (vision — stripped)
        robot_states           (T, 9)   float32  [gripper(2), eef_pos(3), eef_quat(4)]
        states                 (T, 57)  float32  [qpos(30), qvel(27)] — object decode
        actions                (T, 7)   float32

Usage::

    uv run python scripts/make_small_libero_data.py
    uv run python scripts/make_small_libero_data.py --demos 2 --steps 40
    PHASEFORGE_DATA_DIR=/mnt/data uv run python scripts/make_small_libero_data.py

IMPORTANT: the FSM rejects any suite folder that does not contain EXACTLY
the official file count. Before a real download, delete the synthetic files
and the processed cache (see the end of the script).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Ensure the project is on sys.path when run as a plain script.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from phaseforge.data.paths import EXPECTED_FILE_COUNTS, libero_suite_dir  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate_pipeline import (  # noqa: E402
    synthetic_states,
    write_synthetic_object_index,
)

TASK_COUNT = EXPECTED_FILE_COUNTS["libero_90"]
SUITE = "libero_90"
#: All 90 DRYRUN tasks share the same 3-object scene (all free joints).
DRYRUN_OBJECTS = ["cube_0", "cube_1", "cube_2"]


def make_synthetic_libero_file(path: Path, n_demos: int, T: int, seed: int) -> None:
    """Write one HDF5 file matching the real LIBERO flattened schema."""
    rng = np.random.default_rng(seed)
    # Tiny images are fine: the stripper identifies vision keys by NAME and
    # never loads them into RAM (it only needs the keys present in the file).
    H, W = 16, 16

    with h5py.File(path, "w") as f:
        data_grp = f.create_group("data")
        for d in range(n_demos):
            grp = data_grp.create_group(f"demo_{d}")
            obs = grp.create_group("obs")

            # 7-DoF joint states
            obs["joint_states"] = rng.normal(0, 0.5, (T, 7)).astype(np.float32)

            # EE position — cumsum so velocity exceeds the phase-labeler
            # threshold (0.01) and phases are actually detected.
            obs["ee_pos"] = np.cumsum(
                rng.normal(0, 0.02, (T, 3)), axis=0
            ).astype(np.float32)

            # Gripper qpos — the critical phase signal: open -> closed -> open.
            gripper = np.ones((T, 2), np.float32) * 0.06  # "open"
            t1, t2 = T // 3, (2 * T) // 3
            gripper[t1:t2] = 0.005  # "closed" (below the 0.02 threshold)
            obs["gripper_states"] = gripper

            # Vision keys — required by the flattened schema; the stripper
            # must drop them without ever loading them into RAM.
            obs["agentview_rgb"] = np.zeros((T, H, W, 3), dtype=np.uint8)
            obs["eye_in_hand_rgb"] = np.zeros((T, H, W, 3), dtype=np.uint8)

            # robot_states at demo root (9-dim: gripper + eef_pos + eef_quat)
            eef_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (T, 1))
            grp["robot_states"] = np.concatenate(
                [gripper, obs["ee_pos"][:], eef_quat], axis=-1
            ).astype(np.float32)

            # Flattened sim state (P-Stage 1 object-state decode source)
            grp["states"] = synthetic_states(T, rng)

            # 7-dim actions
            grp["actions"] = rng.normal(0, 0.05, (T, 7)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=int, default=3, help="Demos per task")
    parser.add_argument("--steps", type=int, default=60, help="Timesteps per demo")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    args = parser.parse_args()

    suite_dir = libero_suite_dir(SUITE)
    suite_dir.mkdir(parents=True, exist_ok=True)

    existing = list(suite_dir.glob("*.hdf5"))
    if existing and not args.overwrite:
        raise SystemExit(
            f"{suite_dir} already contains {len(existing)} .hdf5 files. "
            f"Re-run with --overwrite to regenerate. The FSM requires exactly "
            f"{TASK_COUNT} files — delete this folder before a real download."
        )

    for i in range(TASK_COUNT):
        p = suite_dir / f"DRYRUN_task_{i:03d}.hdf5"
        make_synthetic_libero_file(p, n_demos=args.demos, T=args.steps, seed=i)

    # P-Stage 1: the FSM loads the object index at
    # {data_root}/raw/libero/object_index.json — every DRYRUN task must be
    # present there, otherwise decoding raises a KeyError on the first file.
    write_synthetic_object_index(
        suite_dir.parent,
        {f"DRYRUN_task_{i:03d}": DRYRUN_OBJECTS for i in range(TASK_COUNT)},
    )

    total_kb = sum(f.stat().st_size for f in suite_dir.glob("*.hdf5")) // 1024
    print(f"Wrote {TASK_COUNT} synthetic tasks to {suite_dir} ({total_kb} KB).")
    print("Next: uv run phaseforge-train models=baselines/bc train=stage1 "
          "project.device=cpu ...")

    print(
        "\nNOTE for the real run: delete this synthetic suite and the "
        "processed cache first, otherwise the FSM integrity check fails and/or "
        "the config-hash cache reuses synthetic normalizer stats:\n"
        "  Remove-Item -Recurse -Force data/raw/libero/libero_90, data/processed"
    )


if __name__ == "__main__":
    main()
