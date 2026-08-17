"""Extract stage-2 warm-start quality and NMI decay per seed/method.

Tests the hypothesis: stage-1 selected-checkpoint quality (action loss at the
SELECTED epoch) propagates through the frozen-encoder warm start into stage-2
routing quality and final rollout success.
"""

import json
import sys
from pathlib import Path

OUTPUTS = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")


def main():
    # (model, seed) -> dict of stage2 info
    stage2 = {}
    for summary in OUTPUTS.rglob("summary.json"):
        try:
            j = json.loads(summary.read_text())
        except Exception:
            continue
        if j.get("kind") != "train" or j.get("stage") != 2:
            continue
        curves = summary.parent / "training_curves.jsonl"
        epoch1 = None
        epochs = []
        if curves.exists():
            for line in curves.read_text().splitlines():
                c = json.loads(line)
                if c["epoch"] == 1:
                    epoch1 = c
                epochs.append(c)
        key = (j.get("model"), j.get("seed"))
        stage2[key] = {
            "run_id": j.get("run_id"),
            "best_epoch": j.get("best_epoch"),
            "best_monitor": j.get("best_val_monitor"),
            "final_act": j.get("final_val", {}).get("loss_action"),
            "final_nmi": j.get("final_val", {}).get("val/phase_expert_nmi"),
            "final_entropy": j.get("final_val", {}).get("val/routing_entropy"),
            "warmstart_act": epoch1.get("val/loss_action") if epoch1 else None,
            "warmstart_nmi": epoch1.get("val/phase_expert_nmi") if epoch1 else None,
            "nmi_epoch10": next(
                (e.get("val/phase_expert_nmi") for e in epochs if e["epoch"] == 10),
                None,
            ),
            "nmi_epoch50": next(
                (e.get("val/phase_expert_nmi") for e in epochs if e["epoch"] == 50),
                None,
            ),
            "nmi_epoch100": next(
                (e.get("val/phase_expert_nmi") for e in epochs if e["epoch"] == 100),
                None,
            ),
            "source": (j.get("source_stage1") or {}).get("model"),
            "freeze": j.get("freeze_encoder"),
        }

    print(f"{'model':32s} {'seed':>4s} {'best_ep':>7s} {'best_mon':>8s} {'warm_act':>9s} "
          f"{'final_act':>9s} {'NMI@1':>6s} {'NMI@10':>6s} {'NMI@50':>6s} {'NMI@100':>6s} "
          f"{'NMI@fin':>7s}")
    for (model, seed) in sorted(stage2):
        d = stage2[(model, seed)]
        print(f"{model:32s} {seed:4d} {d['best_epoch']:7d} {d['best_monitor']:.6f} "
              f"{d['warmstart_act']:.4f} {d['final_act']:.4f} "
              f"{d['warmstart_nmi']:.3f} {d['nmi_epoch10']:.3f} {d['nmi_epoch50']:.3f} "
              f"{d['nmi_epoch100']:.3f} {d['final_nmi']:.3f}")


if __name__ == "__main__":
    main()