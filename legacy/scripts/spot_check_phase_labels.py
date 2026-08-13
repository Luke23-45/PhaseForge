"""C4 gate: spot-check phase labels on real LIBERO trajectories.

Run on the machine that has the mirror (e.g. the Colab box) BEFORE the
bootstrap, so bad labels never reach the phase-supervised training:

    python scripts/spot_check_phase_labels.py --suite libero_90
    python scripts/spot_check_phase_labels.py --suite libero_10 --max-tasks 3

What it does
------------
Reuses the exact production wiring from ``_ingest_and_strip``
(``VisionStripper`` + ``RuleBasedPhaseLabeler`` with slices derived from
the configured ``state_keys`` and the census-built ``ObjectIndex``),
labels a sample of real trajectories, prints per-task phase distributions
and transition counts, and enforces the C4 invariants:

  * labels are int64, length == T, values in [0, num_phases)
  * the sampled corpus has at least one phase transition
  * every phase segment after the first respects ``min_phase_duration``

Exit code 1 on any violation (the bootstrap must not run on bad labels).

Warnings (not failures): phases missing from the corpus — a random demo
may legitimately not complete the full grasp-release cycle; and a task
whose phases are fewer than expected — check it manually.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from phaseforge.data.libero.object_state import ObjectIndex  # noqa: E402
from phaseforge.data.libero.task_index import build_task_index  # noqa: E402
from phaseforge.data.libero.vision_stripper import VisionStripper  # noqa: E402
from phaseforge.data.paths import (  # noqa: E402
    libero_object_index_path,
    libero_suite_dir,
)

_PHASE_NAMES = [
    "APPROACH",
    "PRE_GRASP",
    "GRASP",
    "TRANSPORT",
    "PLACE",
    "RETRACT",
]


def _proprio_slices(cfg) -> tuple[tuple[int, int], tuple[int, int]]:
    """Derive labeler slices from configured state_keys.

    Mirrors ``DataPipelineStateMachine._proprio_slices`` — the proprio
    block is the concatenation of ``state_keys`` in config order; the
    object block and mask are appended after it.
    """
    cursor = 0
    by_key: dict[str, tuple[int, int]] = {}
    for entry in cfg.state_keys:
        key = entry["key"]
        dim = int(entry["dim"])
        by_key[key] = (cursor, cursor + dim)
        cursor += dim
    return by_key["robot0_eef_pos"], by_key["robot0_gripper_qpos"]


def _check_trajectory(
    labels: np.ndarray, num_phases: int, min_phase_duration: int
) -> list[str]:
    """Return the list of C4 invariant violations for one trajectory."""
    errors: list[str] = []
    T = labels.shape[0]
    if labels.dtype != np.int64:
        errors.append(f"labels dtype {labels.dtype} != int64")
    if T == 0:
        return errors
    if labels.min() < 0 or labels.max() >= num_phases:
        errors.append(
            f"labels outside [0, {num_phases}): min={labels.min()} max={labels.max()}"
        )
    changes = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate([[0], changes, [T]])
    for start, end in zip(bounds[:-1], bounds[1:]):
        if start == 0:
            continue
        if end - start < min_phase_duration:
            errors.append(
                f"segment [{start}, {end}) of phase {int(labels[start])} "
                f"shorter than min_phase_duration={min_phase_duration}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_90", help="Suite to spot-check")
    parser.add_argument("--data-root", default=None, help="Data root (default: env/./data)")
    parser.add_argument("--max-tasks", type=int, default=5, help="Tasks to sample")
    parser.add_argument("--max-demos", type=int, default=3, help="Demos per task")
    args = parser.parse_args()

    raw_dir = libero_suite_dir(args.suite, args.data_root)
    if not raw_dir.exists():
        print(f"ERROR: suite dir not found: {raw_dir} (mirror not downloaded?)")
        return 1

    from hydra.utils import instantiate

    common_cfg = OmegaConf.load(REPO / "phaseforge" / "config" / "data" / "common.yaml")
    suite_cfg = OmegaConf.load(
        REPO / "phaseforge" / "config" / "data" / "libero" / f"{args.suite}.yaml"
    )
    pl_cfg = suite_cfg.phase_labeler

    object_index = None
    oscfg = common_cfg.object_state
    if oscfg.enabled:
        raw_path = oscfg.index_path
        idx_path = (
            Path(raw_path) if raw_path else libero_object_index_path(args.data_root)
        )
        object_index = ObjectIndex.load(idx_path)

    eef_slice, gripper_slice = _proprio_slices(common_cfg)
    labeler = instantiate(
        pl_cfg,
        eef_pos_slice=eef_slice,
        gripper_qpos_slice=gripper_slice,
    )
    num_phases = int(pl_cfg.num_phases)
    min_phase_duration = int(pl_cfg.min_phase_duration)

    stripper = VisionStripper(
        state_keys=list(common_cfg.state_keys),
        task_index=build_task_index(raw_dir),
        object_index=object_index,
    )

    errors: list[str] = []
    warnings: list[str] = []
    corpus_transitions = 0
    corpus_phases: Counter = Counter()

    files = sorted(raw_dir.glob("*.hdf5"))[: args.max_tasks]
    if not files:
        print(f"ERROR: no HDF5 files in {raw_dir}")
        return 1
    print(f"Spot-checking {len(files)} task(s) x up to {args.max_demos} demo(s) "
          f"from {raw_dir}\n")

    for hdf5_path in files:
        trajectories = stripper.strip(hdf5_path)
        task_dist: Counter = Counter()
        task_transitions = 0
        n_labeled = 0
        for traj in trajectories[: args.max_demos]:
            labels = labeler.label(traj)
            n_labeled += 1
            task_dist.update(labels.tolist())
            task_transitions += int(np.count_nonzero(np.diff(labels)))
            corpus_phases.update(labels.tolist())
            for msg in _check_trajectory(labels, num_phases, min_phase_duration):
                errors.append(f"{hdf5_path.name}: {msg}")
        corpus_transitions += task_transitions
        if n_labeled == 0:
            warnings.append(f"{hdf5_path.name}: no demos labeled")
            continue
        if task_transitions == 0:
            warnings.append(
                f"{hdf5_path.name}: no phase transitions in the sampled demos"
            )
        dist = ", ".join(
            f"{_PHASE_NAMES[p] if p < len(_PHASE_NAMES) else p}={task_dist[p]}"
            for p in sorted(task_dist)
        )
        missing = set(range(num_phases)) - set(task_dist)
        flag = f"  [MISSING: {sorted(missing)}]" if missing else ""
        print(f"  {hdf5_path.name}: transitions={task_transitions} {dist}{flag}")

    print()
    present = {p for p in corpus_phases}
    missing_all = set(range(num_phases)) - present
    if missing_all:
        warnings.append(f"corpus misses phases {sorted(missing_all)}")
    if corpus_transitions == 0:
        errors.append("no phase transitions anywhere in the sampled corpus")

    print(f"Corpus: {len(files)} task(s), {sum(corpus_phases.values())} labeled "
          f"timesteps, {corpus_transitions} transitions, phases present: "
          f"{sorted(present)}")
    for msg in warnings:
        print(f"WARNING: {msg}")
    if errors:
        print(f"FAILED ({len(errors)} invariant violation(s)):")
        for msg in errors:
            print(f"  - {msg}")
        return 1
    print("C4 gate OK — labels are in range, well-formed, and segmented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
