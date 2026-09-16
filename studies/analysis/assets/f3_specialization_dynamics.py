"""F3 — Stage-2 specialization dynamics: NMI, routing entropy, switch rate.

PhaseForge vs the H1/H2 controls and the scratch floor on Lift and Can;
bold line is seed mean with shaded seed-range ribbon; diamond marker at epoch 0
shows the bootstrap instant (t0 from init_routing.json); single unified top legend.
"""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import plot_seed_trajectories, save

METHODS = (
    "precision_residual_phaseforge",
    "final_aligned_softmax_top1",
    "precision_residual_phase_random_router",
    "precision_residual_plain_encoder",
)
FIELDS = (
    ("nmi", "Phase–Expert NMI", (0.0, 1.05)),
    ("switch_rate", "Routing Switch Rate", (0.0, 0.15)),
    ("routing_entropy", "Normalized Routing Entropy ($H / \\ln K$)", (0.0, 1.05)),
)


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from studies.analysis.stats.trajectories import common_grid, resample

    tasks = [t for t in ("Can", "Square") if t in registry.tasks()]
    method_names = [m for m in METHODS if m in registry.matrix_method_names() or any(m == em.name for em in registry.methods("final"))]
    with paper_style():
        fig, axes = plt.subplots(
            len(FIELDS),
            len(tasks),
            figsize=(7.2, 6.2),
            squeeze=False,
            sharex="col",
        )
        for col, task in enumerate(tasks):
            for row, (field, ylabel, ylim) in enumerate(FIELDS):
                ax = axes[row][col]
                for method in method_names:
                    color = method_color(method)
                    per_seed_series = []
                    t0_markers = []
                    for seed in registry.seeds("final"):
                        key = (task, method, seed, 2)
                        if key in dataset.curves:
                            series = dataset.curves[key].series(field)
                            if series:
                                per_seed_series.append(series)
                        # Extract seed-specific t=0 marker
                        init = dataset.init_routing.get(key)
                        if init is not None:
                            val = None
                            if field == "nmi" and init.t0_nmi is not None:
                                val = float(init.t0_nmi)
                            elif field == "routing_entropy" and init.t0_normalized_routing_entropy is not None:
                                val = float(init.t0_normalized_routing_entropy)
                            if val is not None and not np.isnan(val):
                                t0_markers.append(val)

                    if not per_seed_series:
                        continue

                    # 1. Plot individual seed trajectories as thin, semi-transparent lines
                    for s_data in per_seed_series:
                        xs = [pt[0] for pt in s_data]
                        ys = [pt[1] for pt in s_data]
                        ax.plot(xs, ys, color=color, alpha=0.35, linewidth=0.9, linestyle="--")

                    # 2. Resample and plot arithmetic seed mean as a heavier line
                    grid = common_grid(per_seed_series)
                    if grid:
                        resampled = [resample(s, grid) for s in per_seed_series]
                        matrix = np.asarray(resampled, dtype=float)
                        means = np.nanmean(matrix, axis=0)
                        ax.plot(grid, means, color=color, linewidth=2.0, alpha=1.0,
                                label=registry.display_name(method) if (row == 0 and col == 0) else None,
                                zorder=4)

                    # 3. Plot each validated t=0 value as a seed-level marker
                    for t0_val in t0_markers:
                        ax.scatter([0], [t0_val], facecolors="none", edgecolors=color,
                                   marker="D", s=22, linewidth=1.0, zorder=5)

                ax.set_ylim(ylim)
                ax.grid(True, linestyle=":", alpha=0.35)
                if col == 0:
                    ax.set_ylabel(ylabel, fontsize=8.5)
                if row == len(FIELDS) - 1:
                    ax.set_xlabel("Stage-2 Epoch", fontsize=8.5)
                if row == 0:
                    ax.set_title(task, fontsize=10, fontweight="bold", pad=8)

        # Build clean two-row top legend: methods on top, mark semantics below
        from matplotlib.lines import Line2D
        h_methods, l_methods = axes[0][0].get_legend_handles_labels()
        h_marks = [
            Line2D([0], [0], color="#555555", lw=2.0, label="Seed mean"),
            Line2D([0], [0], color="#888888", lw=0.9, linestyle="--", label="Individual seed trajectory"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="none", markeredgecolor="#555555", markersize=5, label="t=0 init point"),
        ]

        leg1 = fig.legend(
            h_methods,
            l_methods,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=4,
            frameon=False,
            fontsize=8.0,
        )
        leg2 = fig.legend(
            h_marks,
            [m.get_label() for m in h_marks],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=3,
            frameon=False,
            fontsize=7.5,
        )
        fig.add_artist(leg1)
        fig.subplots_adjust(top=0.86, bottom=0.08, left=0.13, right=0.96, hspace=0.28, wspace=0.18)
    return save(fig, "figures/main/F3_specialization")
