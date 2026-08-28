"""F4 — partial warm-start drop-rate sweep and expert initialization diagnostics.

Left panel: Rollout Success Rate vs Expert Parameter Drop Rate (0%, 25%, 50%, 75%, 100%),
with Wilson 95% CIs and individual seed points; canonical R50 (50%) highlighted.
Right panel: Final Phase-Expert NMI across the drop rate sweep.
Reference baselines (Full Warm-Start and One-Warm+Random) plotted with distinct markers.
"""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import OKABE_ITO, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import save
from studies.analysis.stats.intervals import seed_mean_and_wilson

SWEEP = ("pf_drop00", "pf_drop25", "pf_drop50", "pf_drop75", "pf_drop100")
DROP_RATE = {
    "pf_drop00": 0.0,
    "pf_drop25": 0.25,
    "pf_drop50": 0.50,
    "pf_drop75": 0.75,
    "pf_drop100": 1.00,
}
CANONICAL_ALIASES = {"pf_drop50": "phaseforge"}


def _get_eval(dataset: AnalysisDataset, name: str, seed: int):
    k1 = (None, name, seed)
    if k1 in dataset.evals:
        return dataset.evals[k1]
    k2 = ("Lift", name, seed)
    if k2 in dataset.evals:
        return dataset.evals[k2]
    return None


def _get_curve(dataset: AnalysisDataset, name: str, seed: int, stage: int = 2):
    k1 = (None, name, seed, stage)
    if k1 in dataset.curves:
        return dataset.curves[k1]
    k2 = ("Lift", name, seed, stage)
    if k2 in dataset.curves:
        return dataset.curves[k2]
    return None


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    with paper_style():
        fig, (ax_sr, ax_nmi) = plt.subplots(1, 2, figsize=(7.0, 2.7), sharex=True)

        xs, sr_means, sr_los, sr_his = [], [], [], []
        sr_seed_points = []
        nmi_means, nmi_mins, nmi_maxs = [], [], []

        for cell in SWEEP:
            name = CANONICAL_ALIASES.get(cell, cell)
            rates, successes, nmis = [], [], []
            for seed in registry.seeds("ablation"):
                ev = _get_eval(dataset, name, seed)
                if ev is not None:
                    rates.append(ev.success_rate)
                    successes.append(ev.successes)
                curve = _get_curve(dataset, name, seed, 2)
                if curve is not None:
                    n = curve.last("nmi")
                    if n is not None:
                        nmis.append(n)

            if not rates:
                continue

            x = DROP_RATE[cell]
            xs.append(x)
            p, lo, hi = seed_mean_and_wilson(successes, 50)
            sr_means.append(p)
            sr_los.append(lo)
            sr_his.append(hi)
            sr_seed_points.append(rates)

            if nmis:
                nmi_means.append(float(np.mean(nmis)))
                nmi_mins.append(min(nmis))
                nmi_maxs.append(max(nmis))

        # --- Panel A: Success Rate ---
        # Ribbon for Wilson CI
        ax_sr.fill_between(xs, sr_los, sr_his, color=OKABE_ITO["vermillion"], alpha=0.15)
        # Error bars + main line
        yerr = [
            np.array(sr_means) - np.array(sr_los),
            np.array(sr_his) - np.array(sr_means),
        ]
        ax_sr.errorbar(
            xs,
            sr_means,
            yerr=yerr,
            fmt="o-",
            color=OKABE_ITO["vermillion"],
            capsize=3.0,
            linewidth=2.0,
            markersize=6,
            label="Partial warm-start (Wilson 95% CI)",
            zorder=4,
        )

        # Individual seed points
        for x_val, s_list in zip(xs, sr_seed_points):
            for s_val in s_list:
                ax_sr.scatter(
                    [x_val],
                    [s_val],
                    facecolors="none",
                    edgecolors=OKABE_ITO["vermillion"],
                    s=22,
                    linewidths=0.9,
                    alpha=0.75,
                    zorder=5,
                )

        # Reference cells
        full_warm_rates = [
            _get_eval(dataset, "pf_full_warm", s).success_rate
            for s in registry.seeds("ablation")
            if _get_eval(dataset, "pf_full_warm", s) is not None
        ]
        if full_warm_rates:
            m_fw = float(np.mean(full_warm_rates))
            ax_sr.scatter(
                [0.0],
                [m_fw],
                marker="^",
                color=OKABE_ITO["orange"],
                s=55,
                edgecolor="black",
                linewidth=0.8,
                label=f"Full warm-start ({m_fw:.2f})",
                zorder=6,
            )

        onewarm_rates = [
            _get_eval(dataset, "pf_one_warm_plus_random", s).success_rate
            for s in registry.seeds("ablation")
            if _get_eval(dataset, "pf_one_warm_plus_random", s) is not None
        ]
        if onewarm_rates:
            m_ow = float(np.mean(onewarm_rates))
            ax_sr.scatter(
                [0.833],
                [m_ow],
                marker="v",
                color=OKABE_ITO["purple"],
                s=55,
                edgecolor="black",
                linewidth=0.8,
                label=f"One warm + 5 rand ({m_ow:.2f})",
                zorder=6,
            )

        ax_sr.axvline(0.5, color=OKABE_ITO["grey"], linewidth=1.0, linestyle="--", alpha=0.7)
        ax_sr.axvspan(0.46, 0.54, color=OKABE_ITO["vermillion"], alpha=0.08)
        ax_sr.annotate(
            "Canonical (R50)",
            xy=(0.50, 0.72),
            xytext=(0.53, 0.78),
            fontsize=8,
            fontweight="bold",
            color=OKABE_ITO["vermillion"],
            arrowprops=dict(arrowstyle="->", color=OKABE_ITO["vermillion"], lw=1.0),
        )

        ax_sr.set_xlabel("Expert Drop Rate", fontsize=9)
        ax_sr.set_ylabel("Lift Success Rate", fontsize=9)
        ax_sr.set_ylim(0.35, 0.85)
        ax_sr.set_xlim(-0.05, 1.05)
        ax_sr.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax_sr.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax_sr.grid(True, linestyle=":", alpha=0.3)
        ax_sr.legend(frameon=False, fontsize=7.5, loc="lower left")
        ax_sr.set_title("A. Policy Performance vs Drop Rate", fontsize=9.5, fontweight="bold")

        # --- Panel B: Specialization (NMI) ---
        if nmi_means:
            ax_nmi.fill_between(xs, nmi_mins, nmi_maxs, color=OKABE_ITO["blue"], alpha=0.15)
            ax_nmi.plot(
                xs,
                nmi_means,
                "s-",
                color=OKABE_ITO["blue"],
                linewidth=2.0,
                markersize=6,
                label="Final Phase–Expert NMI",
                zorder=4,
            )

        ax_nmi.axvline(0.5, color=OKABE_ITO["grey"], linewidth=1.0, linestyle="--", alpha=0.7)
        ax_nmi.set_xlabel("Expert Drop Rate", fontsize=9)
        ax_nmi.set_ylabel("Final NMI (Stage 2)", fontsize=9)
        ax_nmi.set_ylim(0.30, 0.50)
        ax_nmi.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax_nmi.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax_nmi.grid(True, linestyle=":", alpha=0.3)
        ax_nmi.legend(frameon=False, fontsize=7.5, loc="upper right")
        ax_nmi.set_title("B. Phase Specialization vs Drop Rate", fontsize=9.5, fontweight="bold")

        fig.tight_layout()
    return save(fig, "figures/main/F4_drop_sweep")
