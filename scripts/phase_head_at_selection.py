"""Phase-head state at the monitor-selected checkpoint + plateau flatness per seed."""

from __future__ import annotations

import json
from pathlib import Path

SEEDS = ["42", "43", "44"]


def main() -> None:
    print("Monitor plateau + phase-head quality at selected epoch (stage-1, phaseforge)")
    for seed in SEEDS:
        d = Path("outputs/part1/outputs/phaseforge/stage1") / f"seed{seed}"
        run = sorted(d.iterdir())[0]
        rows = [json.loads(l) for l in (run / "metrics" / "training_curves.jsonl").read_text().splitlines()]
        best = min(rows, key=lambda r: r["val/loss_action"])
        # plateau: all epochs whose val/loss_action is within 2% of the best
        thr = best["val/loss_action"] * 1.02
        plateau = [r["epoch"] for r in rows if r["val/loss_action"] <= thr]
        print(f"  seed{seed}: selected_ep={best['epoch']} best_val_action={best['val/loss_action']:.4f} "
              f"plateau_eps=[{plateau[0]}-{plateau[-1]}] (n={len(plateau)})")
        print(f"      at selected: phase_loss={best['val/loss_phase']:.3f} phase_acc={best.get('val/phase_acc')}")
        print(f"      ep1:         phase_loss={rows[0]['val/loss_phase']:.3f} phase_acc={rows[0].get('val/phase_acc')}")
        print(f"      final:       phase_loss={rows[-1]['val/loss_phase']:.3f} phase_acc={rows[-1].get('val/phase_acc')}")


if __name__ == "__main__":
    main()