"""F2 — per-task paired success deltas (PhaseForge vs key baselines/controls).

1x4 horizontal strip forest plot; rows are tasks; point is the paired seed-mean delta
on identical reset cases; intervals span min-max seed deltas with distinct caps;
individual seed deltas are vertically jittered (n=3) to prevent overplotting.
"""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import forest, save
from studies.analysis.stats.paired import pair_episodes

COMPARISONS = (
    "bc",
    "final_aligned_softmax_top1",
    "precision_residual_phase_random_router",
    "precision_residual_plain_encoder",
)
PANEL_TITLES = {
    "bc": "vs. BC",
    "final_aligned_softmax_top1": "vs. Softmax Top-1",
    "precision_residual_phase_random_router": "vs. Phase-Random",
    "precision_residual_plain_encoder": "vs. Plain Encoder",
}


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = list(reversed(registry.tasks()))  # Lift at top
    pf_name = "precision_residual_phaseforge" if (tasks[0], "precision_residual_phaseforge", 42) in dataset.evals else "phaseforge"
    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharey=True, sharex=True)
        comparator_grid = [
            [("bc", axes[0, 0]), ("final_aligned_softmax_top1", axes[0, 1])],
            [("precision_residual_phase_random_router", axes[1, 0]), ("precision_residual_plain_encoder", axes[1, 1])],
        ]

        for r_idx, row in enumerate(comparator_grid):
            for c_idx, (comparator, ax) in enumerate(row):
                labels, means, lows, highs, seed_points = [], [], [], [], []
                for task in tasks:
                    seed_deltas = []
                    for seed in registry.seeds("final"):
                        key_a = (task, pf_name, seed)
                        key_b = (task, comparator, seed)
                        if key_a not in dataset.episodes or key_b not in dataset.episodes:
                            continue
                        bank_a = dataset.evals[key_a].reset_bank
                        bank_b = dataset.evals[key_b].reset_bank
                        if bank_a != bank_b:
                            continue  # pairing invalid across different banks
                        outcome = pair_episodes(
                            task,
                            seed,
                            dataset.episodes[key_a],
                            dataset.episodes[key_b],
                            bank_a=bank_a,
                            bank_b=bank_b,
                        )
                        seed_deltas.append(outcome.delta)
                    if not seed_deltas:
                        continue
                    labels.append(task)
                    mean_delta = float(np.mean(seed_deltas))
                    means.append(mean_delta)
                    lows.append(min(seed_deltas))
                    highs.append(max(seed_deltas))
                    seed_points.append(seed_deltas)

                color = method_color(comparator)
                forest(
                    ax,
                    labels,
                    means,
                    lows,
                    highs,
                    colors=[color] * len(labels),
                    seed_points=seed_points,
                    xlabel="",
                    show_zero=True,
                    capsize=3.0,
                )
                ax.set_title(PANEL_TITLES[comparator], fontsize=9.5, fontweight="bold", pad=5)
                ax.set_xlim(-0.52, 0.52)
                ax.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
                ax.grid(axis="x", linestyle=":", alpha=0.35)

        # Single clean centered xlabel at bottom
        fig.text(
            0.55, 0.025,
            r"Paired Difference: PhaseForge − Comparator $\Delta$ (Observed Seed Range)",
            ha="center", fontsize=8.5, fontweight="bold"
        )

        # Unified legend for mark semantics
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#444444", markersize=6, label="Seed mean paired Δ"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="#444444", markersize=5, label="Individual seed paired Δ (n=3)"),
            Line2D([0], [0], color="#444444", lw=1.8, label="Observed seed range [min, max]"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=3,
            frameon=False,
            fontsize=8.0,
        )

        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.14, right=0.96, hspace=0.36, wspace=0.18)
    return save(fig, "figures/main/F2_paired_deltas")
