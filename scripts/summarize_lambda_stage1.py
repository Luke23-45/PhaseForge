"""Summarize the λ-decay stage-1 runs: phase-loss trajectory + plateau shape."""

from __future__ import annotations

import glob
import json

ROOT = "outputs_local_train/phaseforge/stage1"


def main() -> None:
    for s in ("42", "43", "44"):
        path = glob.glob(f"{ROOT}/seed{s}/*lambdav1*/metrics/training_curves.jsonl")[0]
        rows = [json.loads(line) for line in open(path)]
        phases = [r["val/loss_phase"] for r in rows]
        best = min(rows, key=lambda r: r["val/loss_action"])
        thr = best["val/loss_action"] * 1.02
        plateau = [r["epoch"] for r in rows if r["val/loss_action"] <= thr]
        print(
            f"seed{s}: val/loss_phase min={min(phases):.3f}@{phases.index(min(phases))} "
            f"max={max(phases):.3f} final={phases[-1]:.3f} | "
            f"best_ep={best['epoch']} best_action={best['val/loss_action']:.4f} "
            f"plateau=[{plateau[0]}-{plateau[-1]}] n={len(plateau)}"
        )


if __name__ == "__main__":
    main()