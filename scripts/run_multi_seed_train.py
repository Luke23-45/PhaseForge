"""Run full multi-seed training for all PhaseForge model variants.

DEPRECATED — superseded by ``phaseforge-sweep`` (see
``docs/plan/experiment_runner.md``). This helper does NOT cover the
``bc_robot_only`` cell (``data=robot_only``) and does NOT run the offline
evaluation steps; it is kept only for reference. Use the runner instead:

    phaseforge-sweep --list          # show the full method matrix
    phaseforge-sweep --stage 1       # train stage 1 for all methods/seeds
    phaseforge-sweep --with-dependencies   # full sweep incl. eval

Trains every model for every seed so that downstream evaluation can report
mean +/- std over training seeds. Stage 2 runs auto-detect the SAME seed's
Stage 1 checkpoint via seed-matched lookup
(``phaseforge.utils.config.find_latest_checkpoint``), so each seed trains an
independent, consistent pipeline.

Cell inventory (issues register C1/C7):
    * Stage 1 only:  bc (plain), phaseforge (phase-supervised).
    * Stage 2 only:  scratch_moe, oracle_moe, warmstart_moe (plain encoder
      x random router), phase_pretrain_random_router (phase encoder x
      random router), plain_encoder_phase_bootstrap (plain encoder x
      centroid router), teacher_forced (E8: GT-partitioned experts,
      predicted-phase routing at inference).

Usage:
    uv run python scripts/run_multi_seed_train.py

Requires:
    - Real robomimic low-dimensional data cache built by the data pipeline before the first
      training run (the pipeline prints the config hash; stage-2 lookups
      are seed-matched automatically).
    - CUDA machine (run on the GPU box, not a CPU laptop).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Model config -> list of training stages to run (BC is Stage 1 only).
#
# The 2x2 factorial cells (C1) and the teacher-forced cell (E8) run Stage 2
# only: their Stage 1 checkpoints are shared (resolve_checkpoint_source)
# with ``phaseforge`` (phase-supervised) or ``bc`` (plain) — which must be
# trained first, hence the dict ordering below.
MODEL_STAGES: dict[str, tuple[str, list[int]]] = {
    "bc": ("baselines/bc", [1]),
    "phaseforge": ("phaseforge", [1, 2]),
    "scratch_moe": ("baselines/scratch_moe", [2]),
    "warmstart_moe": ("baselines/warmstart_moe", [2]),
    "oracle_moe": ("baselines/oracle_moe", [2]),
    "phase_pretrain_random_router": ("baselines/phase_pretrain_random_router", [2]),
    "plain_encoder_phase_bootstrap": ("baselines/plain_encoder_phase_bootstrap", [2]),
    "teacher_forced": ("baselines/teacher_forced", [2]),
}

SEEDS = [42, 43, 44]


def run_train(model_cfg: str, stage: int, seed: int) -> None:
    cmd = [
        "phaseforge-train",
        f"models={model_cfg}",
        f"train=stage{stage}",
        f"project.seed={seed}",
        # Locked protocol (research_definition.md §5): full-length schedules
        # with NO truncating early stop. The stage yamls default
        # early_stopping to enabled (patience 10), which would silently
        # truncate the protocol runs — disable it explicitly.
        "train.early_stopping.enabled=false",
    ]
    print(
        f"[seed {seed}] START model={model_cfg} stage={stage}",
        flush=True,
    )
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"[seed {seed}] FAILED model={model_cfg} stage={stage} "
            f"(return code {result.returncode})",
            flush=True,
        )
        sys.exit(result.returncode or 1)
    print(
        f"[seed {seed}] COMPLETE model={model_cfg} stage={stage}",
        flush=True,
    )


def print_ledger_summary() -> None:
    """Print per-model/stage run status counts from the outputs ledger."""
    # Anchored to the default `project.output_dir` ("outputs"); keep in sync
    # if the sweep runs with a different override. The existence check is
    # deliberate: constructing a RunLedger would create the directory.
    ledger_dir = Path("outputs") / "_ledger"
    if not (ledger_dir / "runs.jsonl").exists():
        print("[ledger] no runs recorded yet.")
        return
    from phaseforge.outputs_writer.ledger import RunLedger

    rows = RunLedger(ledger_dir).read_all()
    if not rows:
        print("[ledger] ledger is empty.")
        return
    summary: dict[tuple[str, int | None], dict[str, int]] = {}
    for row in rows:
        counts = summary.setdefault((row.model, row.stage), {})
        counts[row.status] = counts.get(row.status, 0) + 1
    print("\n[ledger] per model/stage run status:")
    for (model, stage), counts in sorted(summary.items()):
        print(
            f"  {model:<38} stage {stage}: "
            + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        )


def main() -> None:
    for seed_index, seed in enumerate(SEEDS, start=1):
        print(
            f"\n[multi-seed {seed_index}/{len(SEEDS)}] START seed={seed}",
            flush=True,
        )
        for model_name, (model_cfg, stages) in MODEL_STAGES.items():
            print(f"[seed {seed}] MODEL {model_name}", flush=True)
            for stage in stages:
                run_train(model_cfg, stage, seed)
        if seed_index < len(SEEDS):
            next_seed = SEEDS[seed_index]
            print(
                f"[multi-seed] COMPLETE seed={seed}; starting next seed={next_seed}",
                flush=True,
            )
        else:
            print(f"[multi-seed] COMPLETE seed={seed}", flush=True)
    print("[multi-seed] ALL SEEDS COMPLETE", flush=True)
    print_ledger_summary()


if __name__ == "__main__":
    main()
