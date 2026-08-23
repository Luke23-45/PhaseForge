"""F4 — partial warm-start drop-rate sweep (SR left, final NMI right).

Canonical R50 (drop 50%) annotated; full-warm and one-warm-plus-random drawn
as reference markers at their effective positions.
"""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import OKABE_ITO, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import save
from studies.analysis.stats.intervals import mean

SWEEP = ("pf_drop00", "pf_drop25", "pf_drop50", "pf_drop75", "pf_drop100")
DROP_RATE = {
    "pf_drop00": 0.0,
    "pf_drop25": 0.25,
    "pf_drop50": 0.5,
    "pf_drop75": 0.75,
    "pf_drop100": 1.0,
}
# pf_drop50 is the canonical phaseforge row in the ablation namespace.
CANONICAL_ALIASES = {"pf_drop50": "phaseforge"}


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    with paper_style():
        fig, ax_sr = plt.subplots(figsize=(4.6, 2.9))
        ax_nmi = ax_sr.twinx()
        ax_nmi.spines["right"].set_visible(True)

        xs_sr, ys_sr, lo_sr, hi_sr = [], [], [], []
        xs_nmi, ys_nmi = [], []
        for cell in SWEEP:
            name = CANONICAL_ALIASES.get(cell, cell)
            rates, nmis = [], []
            for seed in registry.seeds("ablation"):
                key = (None, name, seed)
                if key not in dataset.evals:
                    continue
                rates.append(dataset.evals[key].success_rate)
                curve_key = (None, name, seed, 2)
                if curve_key in dataset.curves:
                    nmi = dataset.curves[curve_key].last("nmi")
                    if nmi is not None:
                        nmis.append(nmi)
            if not rates:
                continue
            x = DROP_RATE[cell]
            xs_sr.append(x)
            ys_sr.append(mean(rates))
            lo_sr.append(min(rates))
            hi_sr.append(max(rates))
            if nmis:
                xs_nmi.append(x)
                ys_nmi.append(mean(nmis))

        ax_sr.fill_between(xs_sr, lo_sr, hi_sr, color=OKABE_ITO["vermillion"], alpha=0.18)
        ax_sr.plot(
            xs_sr, ys_sr, "o-", color=OKABE_ITO["vermillion"], label="success rate (seed range)"
        )
        ax_sr.set_xlabel("partial warm-start drop rate")
        ax_sr.set_ylabel("success rate")
        ax_sr.set_ylim(0, 1)
        if xs_nmi:
            ax_nmi.plot(
                xs_nmi, ys_nmi, "s--", color=OKABE_ITO["blue"], label="final phase–expert NMI"
            )
            ax_nmi.set_ylabel("final NMI")

        # Reference cells (not part of the drop axis): full warm + one warm.
        for cell, marker, label in (
            ("pf_full_warm", "^", "full warm-start"),
            ("pf_one_warm_plus_random", "v", "one warm + random"),
        ):
            rates = [
                dataset.evals[(None, cell, s)].success_rate
                for s in registry.seeds("ablation")
                if (None, cell, s) in dataset.evals
            ]
            if rates:
                ax_sr.scatter(
                    [0.5],
                    [mean(rates)],
                    marker=marker,
                    facecolors="none",
                    edgecolors=OKABE_ITO["black"],
                    s=42,
                    label=f"{label} (ref. @50%)",
                )
        ax_sr.axvline(0.5, color=OKABE_ITO["grey"], linewidth=0.8, linestyle=":")
        ax_sr.annotate(
            "canonical R50", xy=(0.5, 1.0), xytext=(0.52, 1.02), fontsize=8, color=OKABE_ITO["grey"]
        )
        lines1, labels1 = ax_sr.get_legend_handles_labels()
        lines2, labels2 = ax_nmi.get_legend_handles_labels()
        ax_sr.legend(
            lines1 + lines2, labels1 + labels2, frameon=False, fontsize=7, loc="lower center"
        )
        fig.tight_layout()
    return save(fig, "figures/main/F4_drop_sweep")
