"""Quantify stage-1 checkpoint quality at selection time across seeds/methods.

The stage-2 MoE bootstraps centroids from the stage-1 encoder and then
FREEZES it. So the quality of the SELECTED stage-1 checkpoint should predict
stage-2 routing quality and rollout success.
"""

import json
import sys
from pathlib import Path

OUTPUTS = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")


def find_stage1():
    runs = []
    for summary in OUTPUTS.rglob("summary.json"):
        try:
            j = json.loads(summary.read_text())
        except Exception:
            continue
        if j.get("kind") != "train" or j.get("stage") != 1:
            continue
        curves = summary.parent / "training_curves.jsonl"
        best_epoch = j.get("best_epoch")
        monitor_at_best = None
        if curves.exists():
            for line in curves.read_text().splitlines():
                c = json.loads(line)
                if c["epoch"] == best_epoch:
                    monitor_at_best = c.get("val/loss_action")
        runs.append(
            (
                j.get("model", "?"),
                j.get("seed"),
                j.get("run_id"),
                j.get("best_epoch"),
                j.get("best_val_monitor"),
                monitor_at_best,
                j.get("final_val", {}).get("loss_action"),
            )
        )
    return runs


def main():
    runs = find_stage1()
    by_model = {}
    for model, seed, run_id, best_ep, monitor, act_at_best, final_act in sorted(runs):
        by_model.setdefault(model, []).append(
            (seed, run_id, best_ep, monitor, act_at_best, final_act)
        )
    for model in sorted(by_model):
        print(f"== {model} ==")
        for seed, run_id, best_ep, monitor, act_at_best, final_act in by_model[model]:
            print(
                f"  seed{seed} run={run_id} best_ep={best_ep} "
                f"best_monitor={monitor:.4f} loss_action@best={act_at_best} "
                f"final_act={final_act:.4f}"
            )


if __name__ == "__main__":
    main()