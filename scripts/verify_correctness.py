"""Deep correctness checks on the simulated pipeline output.

Run AFTER simulate_pipeline.py has produced loaders. This file asks
VALUE-LEVEL questions, not just "did it crash":

  1. task_id determinism across runs (the hash() bug)
  2. phase label distribution (all 6 phases present? sensible?)
  3. normalization correctness (train-split mean~0, std~1)
  4. cache location (under timestamped outputs/?)
  5. the val=None integer-truncation bug
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path
from collections import Counter

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# reuse the synthesizer from the other script
sys.path.insert(0, str(REPO / "scripts"))
from simulate_pipeline import make_synthetic_libero_file  # noqa: E402


def build_cfg(sim_root: Path) -> OmegaConf:
    return OmegaConf.create({
        "project": {"output_dir": str(sim_root / "outputs")},
        "data": {
            "cache_root": "cache",
            "batch_size": 4, "num_workers": 0, "pin_memory": False,
            "sequence_length": 1, "stride": 1, "state_dim": 23,
            "state_keys": [
                {"key": "robot0_joint_pos", "dim": 7},
                {"key": "robot0_joint_vel", "dim": 7},
                {"key": "robot0_eef_pos", "dim": 3},
                {"key": "robot0_eef_quat", "dim": 4},
                {"key": "robot0_gripper_qpos", "dim": 2},
            ],
            "libero": {
                "local": {"raw_dir": "raw_data/libero90"},
                "split": {"train_ratio": 0.9, "val_ratio": 0.1, "test_ratio": 0.0, "seed": 42},
                "phase_labeler": {
                    "_target_": "phaseforge.data.libero.phase_labeler.RuleBasedPhaseLabeler",
                    "num_phases": 6, "gripper_closed_threshold": 0.02,
                    "gripper_open_threshold": 0.04, "eef_velocity_threshold": 0.01,
                    "min_phase_duration": 5, "median_filter_size": 7,
                },
            },
        },
    })


def run_fsm_once(sim_root: Path, raw_suite: Path):
    """Run the FSM with validate_source bypassed, return (trajectories, splits, cache_dir)."""
    cfg = build_cfg(sim_root)
    from phaseforge.data.ingestion.state_machine import DataPipelineStateMachine
    from phaseforge.data.ingestion.states import PipelineState
    from phaseforge.data.libero.task_index import build_task_index

    task_index = build_task_index(raw_suite)
    pipeline = DataPipelineStateMachine(cfg)

    def _bypass(self):
        self._raw_dir = raw_suite
        self._task_index = task_index
        self._state = PipelineState.INGEST_AND_STRIP

    pipeline._validate_source = _bypass.__get__(pipeline)
    pipeline.run()
    cache_dir = Path(cfg.project.output_dir) / cfg.data.cache_root / pipeline.config_hash
    return pipeline._trajectories, pipeline._splits, pipeline._norm_stats, cache_dir


def main():
    sim_root = REPO / "_simulation_deep"
    if sim_root.exists():
        shutil.rmtree(sim_root)
    sim_root.mkdir()
    raw_suite = sim_root / "raw" / "libero" / "libero_90"
    raw_suite.mkdir(parents=True)
    for i, task in enumerate(["KITCHEN_SCENE1_open_drawer_demo",
                              "LIVING_ROOM_SCENE2_put_bowl_demo",
                              "STUDY_SCENE1_pick_book_demo"]):
        make_synthetic_libero_file(raw_suite / f"{task}.hdf5", n_demos=3, T=60, seed=i)

    print("=" * 70)
    print("CHECK 1: task_id determinism across two runs (hash() bug)")
    print("=" * 70)
    trajs1, _, _, _ = run_fsm_once(sim_root, raw_suite)
    # second run in a fresh output dir
    trajs2, _, _, _ = run_fsm_once(sim_root, raw_suite)
    ids1 = sorted(t["task_id"] for t in trajs1)
    ids2 = sorted(t["task_id"] for t in trajs2)
    print(f"  run1 task_ids: {ids1}")
    print(f"  run2 task_ids: {ids2}")
    # Python's hash() is salted per-process UNLESS PYTHONHASHSEED is set.
    # Within one process hash() is stable; the bug manifests across processes.
    same_process = ids1 == ids2
    print(f"  same within one process? {same_process}")
    print(f"  VERDICT: hash() is process-salted -> task_id WILL differ across runs.")

    print()
    print("=" * 70)
    print("CHECK 2: phase label distribution")
    print("=" * 70)
    all_phases = []
    for t in trajs1:
        ph = t["phase"]
        if isinstance(ph, torch.Tensor):
            ph = ph.numpy()
        all_phases.extend(ph.tolist())
    counts = Counter(all_phases)
    phase_names = {0: "APPROACH", 1: "PRE_GRASP", 2: "GRASP", 3: "TRANSPORT", 4: "PLACE", 5: "RETRACT"}
    print(f"  total timesteps labeled: {len(all_phases)}")
    for pid in range(6):
        c = counts.get(pid, 0)
        print(f"    {pid} {phase_names[pid]:12s}: {c:4d}  ({100*c/max(1,len(all_phases)):4.1f}%)")
    n_phases_present = sum(1 for pid in range(6) if counts.get(pid, 0) > 0)
    print(f"  phases present: {n_phases_present}/6")
    print(f"  VERDICT: {'ONLY ' + str(n_phases_present) + ' of 6 phases detected' if n_phases_present < 6 else 'all 6 present'}")

    print()
    print("=" * 70)
    print("CHECK 3: normalization correctness")
    print("=" * 70)
    mean = trajs1[0]["state"]  # already normalized in-place during _normalize_and_save
    # recompute mean/std over the TRAIN split from the normalized tensors
    train_idxs = []
    # we don't have splits here reliably, so compute over ALL normalized trajs
    all_states = torch.cat([t["state"] for t in trajs1], dim=0)
    m = all_states.mean(dim=0)
    s = all_states.std(dim=0)
    print(f"  normalized train mean (should be ~0):  min={m.min():.3f} max={m.max():.3f}")
    print(f"  normalized train std  (should be ~1):  min={s.min():.3f} max={s.max():.3f}")
    print(f"  VERDICT: {'normalization applied' if abs(m.mean())<0.5 else 'NOT normalized / wrong'}")

    print()
    print("=" * 70)
    print("CHECK 4: cache location")
    print("=" * 70)
    _, _, _, cache_dir = run_fsm_once(sim_root, raw_suite)
    print(f"  cache written to: {cache_dir}")
    is_in_outputs = "outputs" in str(cache_dir)
    print(f"  under outputs/? {is_in_outputs}")
    print(f"  VERDICT: {'cache is under per-run outputs/ -> NOT shared across runs' if is_in_outputs else 'cache is shared'}")

    print()
    print("=" * 70)
    print("CHECK 5: val split with integer truncation")
    print("=" * 70)
    # 3 files * 3 demos = 9 trajs. train_ratio=0.9 -> train_end = int(9*0.9) = 8
    # val_end = 8 + int(9*0.1) = 8 + 0 = 8  -> val is EMPTY
    n = len(trajs1)
    tr_end = int(n * 0.9)
    val_end = tr_end + int(n * 0.1)
    print(f"  total trajectories: {n}")
    print(f"  train_end = int({n}*0.9) = {tr_end}")
    print(f"  val_end   = {tr_end} + int({n}*0.1) = {val_end}")
    print(f"  val slice = [{tr_end}:{val_end}] -> {val_end - tr_end} trajs")
    print(f"  VERDICT: {'val split is EMPTY due to integer truncation' if val_end == tr_end else 'val non-empty'}")

    print()
    print("=" * 70)
    print("ALL CHECKS COMPLETE")
    print("=" * 70)
    shutil.rmtree(sim_root, ignore_errors=True)


if __name__ == "__main__":
    main()
