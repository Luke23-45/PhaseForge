"""Print stage-2 routing outcomes for one tag across the 3 protocol seeds."""

from __future__ import annotations

import glob
import json
import sys

SEEDS = ("42", "43", "44")
ROOT = "outputs_local_train/phaseforge/stage2"


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "lambdav1_stage2"
    nmis: list[float] = []
    for s in SEEDS:
        paths = [p for p in glob.glob(f"{ROOT}/seed{s}/*/metrics/summary.json") if tag in p]
        if not paths:
            print(f"seed{s}: (no run for tag {tag!r})")
            continue
        j = json.load(open(paths[0]))
        fv = j["final_val"]
        nmis.append(fv["val/phase_expert_nmi"])
        print(
            f"seed{s}: NMI={fv['val/phase_expert_nmi']:.3f} "
            f"entropy={fv['val/routing_entropy']:.3f} "
            f"top1_collapse={fv['val/top1_collapse_rate']} "
            f"topk_collapse={fv['val/topk_collapse_rate']} "
            f"final_action={fv['loss_action']:.4f} best_ep={j['best_epoch']}"
        )
    if len(nmis) == 3:
        print(f"NMI spread: {max(nmis) - min(nmis):.3f}")


if __name__ == "__main__":
    main()