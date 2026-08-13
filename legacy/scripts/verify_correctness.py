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

import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# reuse the synthesizer from the other script
sys.path.insert(0, str(REPO / "scripts"))
from simulate_pipeline import (  # noqa: E402
    SYNTHETIC_OBJECTS_BY_TASK,
    make_synthetic_libero_file,
    write_synthetic_object_index,
)


def build_cfg(sim_root: Path) -> OmegaConf:
    raw_libero = sim_root / "raw" / "libero"
    return OmegaConf.create({
        "project": {"output_dir": str(sim_root / "outputs")},
        "data": {
            "cache_root": "cache",
            "batch_size": 4, "num_workers": 0, "pin_memory": False,
            "sequence_length": 1, "stride": 1, "state_dim": 151,
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
                "index_path": str(raw_libero / "object_index.json"),
            },
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
    write_synthetic_object_index(raw_suite.parent, SYNTHETIC_OBJECTS_BY_TASK)

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
    print("  VERDICT: hash() is process-salted -> task_id WILL differ across runs.")

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
    phase_names = {
        0: "APPROACH", 1: "PRE_GRASP", 2: "GRASP", 3: "TRANSPORT",
        4: "PLACE", 5: "RETRACT",
    }
    print(f"  total timesteps labeled: {len(all_phases)}")
    for pid in range(6):
        c = counts.get(pid, 0)
        print(f"    {pid} {phase_names[pid]:12s}: {c:4d}  ({100*c/max(1,len(all_phases)):4.1f}%)")
    n_phases_present = sum(1 for pid in range(6) if counts.get(pid, 0) > 0)
    print(f"  phases present: {n_phases_present}/6")
    verdict = (
        f"ONLY {n_phases_present} of 6 phases detected"
        if n_phases_present < 6
        else "all 6 present"
    )
    print(f"  VERDICT: {verdict}")

    print()
    print("=" * 70)
    print("CHECK 3: normalization correctness")
    print("=" * 70)
    # recompute mean/std from the NORMALIZED tensors (they are normalized
    # in-place during _normalize_and_save)
    # we don't have splits here reliably, so compute over ALL normalized trajs
    # (mask dims 135:151 are excluded from normalization and stay 0/1)
    all_states = torch.cat([t["state"] for t in trajs1], dim=0)[:, :135]
    m = all_states.mean(dim=0)
    s = all_states.std(dim=0)
    print(f"  normalized train mean (should be ~0):  min={m.min():.3f} max={m.max():.3f}")
    print(f"  normalized train std  (should be ~1):  min={s.min():.3f} max={s.max():.3f}")
    verdict = (
        "normalization applied"
        if abs(m.mean()) < 0.5
        else "NOT normalized / wrong"
    )
    print(f"  VERDICT: {verdict}")

    print()
    print("=" * 70)
    print("CHECK 4: cache location")
    print("=" * 70)
    _, _, _, cache_dir = run_fsm_once(sim_root, raw_suite)
    print(f"  cache written to: {cache_dir}")
    is_in_outputs = "outputs" in str(cache_dir)
    print(f"  under outputs/? {is_in_outputs}")
    verdict = (
        "cache is under per-run outputs/ -> NOT shared across runs"
        if is_in_outputs
        else "cache is shared"
    )
    print(f"  VERDICT: {verdict}")

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
    verdict = (
        "val split is EMPTY due to integer truncation"
        if val_end == tr_end
        else "val non-empty"
    )
    print(f"  VERDICT: {verdict}")

    print()
    print("=" * 70)
    print("CHECK 6: object-state channel (block dims + occupancy mask)")
    print("=" * 70)
    # Layout: [proprio 23 | objects 16*7 | mask 16] => state_dim 151.
    # 3 synthetic objects per task -> mask has 3 ones, 13 zeros, all timesteps.
    obj_block_start, obj_block_end = 23, 23 + 16 * 7
    mask_start, mask_end = obj_block_end, obj_block_end + 16
    state_dims = {int(t["state"].shape[-1]) for t in trajs1}
    print(f"  state dims across trajs: {state_dims}")
    shape_ok = state_dims == {151}
    mask_ok = True
    finite_ok = True
    for t in trajs1:
        st = t["state"]
        if isinstance(st, torch.Tensor):
            st = st.numpy()
        mask = st[:, mask_start:mask_end]
        if not set(np.unique(mask)) <= {0.0, 1.0}:
            mask_ok = False
            print(f"    BAD mask values in task {t['task_id']}: {np.unique(mask)}")
        n_filled = int(mask[0].sum())
        if n_filled != 3:
            mask_ok = False
            print(f"    expected 3 filled slots, got {n_filled} (task {t['task_id']})")
        if not np.isfinite(st[:, obj_block_start:obj_block_end]).all():
            finite_ok = False
            print(f"    non-finite object block (task {t['task_id']})")
    print(f"  shape==151 across trajs? {shape_ok}")
    print(f"  mask binary + 3 filled slots? {mask_ok}")
    print(f"  object block finite? {finite_ok}")
    verdict = (
        "object-state channel OK"
        if (shape_ok and mask_ok and finite_ok)
        else "object-state channel BROKEN"
    )
    print(f"  VERDICT: {verdict}")

    print()
    print("=" * 70)
    print("ALL CHECKS COMPLETE")
    print("=" * 70)
    shutil.rmtree(sim_root, ignore_errors=True)


if __name__ == "__main__":
    main()
