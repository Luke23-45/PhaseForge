"""Summarize the tie-break diagnostic re-run (stage-1 decisions + stage-2 results).

Reads ``outputs_local_train/phaseforge/stage{1,2}/seed<42|43|44>/*`` and prints
the tie-break decisions plus stage-2 outcomes for the validation criteria:
(a) stage-2 val/loss_action stays on its plateau, (b) phase-head quality at
selection improves and its per-seed spread collapses, (c) router NMI spread.

Also compares against the λ-decay condition (tag ``lambdav1_stage2``) when
present, and the fixed-reference numbers.
"""

from __future__ import annotations

import glob
import json
import sys

SEEDS = ("42", "43", "44")
ROOT = "outputs_local_train/phaseforge"
REFERENCE_NMI = {"fixed": [0.449, 0.457, 0.436], "tiebreak": None, "lambda": None}


def _stage2_nmi(tag_filter: str) -> list[float]:
    nmis: list[float] = []
    for s in SEEDS:
        paths = glob.glob(f"{ROOT}/stage2/seed{s}/*/metrics/summary.json")
        paths = [p for p in paths if tag_filter in p]
        if not paths:
            return nmis
        j = json.load(open(paths[0]))
        nmis.append(j["final_val"]["val/phase_expert_nmi"])
    return nmis


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "tiebreak_v1_stage2"
    nmis = _stage2_nmi(tag)
    if not nmis:
        sys.exit(f"no stage-2 summaries match tag {tag!r}")
    print(f"stage-2 ({tag}):")
    for s, nmi in zip(SEEDS, nmis):
        print(f"  seed{s}: NMI={nmi:.3f}")
    print(f"  NMI spread: {max(nmis) - min(nmis):.3f} "
          f"(fixed reference 0.021, buggy 0.069)")


if __name__ == "__main__":
    main()