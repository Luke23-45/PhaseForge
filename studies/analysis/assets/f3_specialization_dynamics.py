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
    "phaseforge",
    "phase_pretrain_random_router",
    "plain_encoder_phase_bootstrap",
    "scratch_moe",
)
FIELDS = (
    ("nmi", "phase–expert NMI", (0.08, 0.52)),
    ("routing_entropy", "routing entropy", (0.72, 1.00)),
    ("switch_rate", "switch rate", (0.035, 0.125)),
)


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [t for t in ("Lift", "Can") if t in registry.tasks()]
    method_names = [m for m in METHODS if m in registry.matrix_method_names()]
    with paper_style():
        fig, axes = plt.subplots(
            len(FIELDS),
            len(tasks),
            figsize=(7.0, 5.6),
            squeeze=False,
            sharex="col",
        )
        legend_handles, legend_labels = [], []
        for col, task in enumerate(tasks):
            for row, (field, ylabel, ylim) in enumerate(FIELDS):
                ax = axes[row][col]
                for method in method_names:
                    per_seed = []
                    for seed in registry.seeds("final"):
                        key = (task, method, seed, 2)
                        if key in dataset.curves:
                            series = dataset.curves[key].series(field)
                            if series:
                                per_seed.append(series)
                    if not per_seed:
                        continue

                    # Extract t0 marker
                    t0_val = None
                    init = dataset.init_routing.get((task, method, registry.seeds("final")[0], 2))
                    if init is not None:
                        if field == "nmi" and init.t0_nmi is not None:
                            t0_val = float(init.t0_nmi)
                        elif field == "routing_entropy" and init.t0_normalized_routing_entropy is not None:
                            t0_val = float(init.t0_normalized_routing_entropy)

                    plot_seed_trajectories(
                        ax,
                        per_seed,
                        method_color(method),
                        label=registry.display_name(method) if (row == 0 and col == 0) else None,
                        ylabel=ylabel if col == 0 else "",
                        xlabel="Stage-2 epoch" if row == len(FIELDS) - 1 else "",
                        show_ribbon=True,
                        t0_marker=t0_val,
                    )

                ax.set_ylim(ylim)
                ax.grid(True, linestyle=":", alpha=0.3)
                if col == 0:
                    ax.set_ylabel(ylabel, fontsize=9)
                if row == len(FIELDS) - 1:
                    ax.set_xlabel("Stage-2 epoch", fontsize=9)
                if row == 0:
                    ax.set_title(task, fontsize=10, fontweight="bold", pad=8)

        # Collect handles for unified legend
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(
            h,
            l,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=4,
            frameon=False,
            fontsize=8.5,
        )
        fig.subplots_adjust(top=0.88, bottom=0.08, left=0.10, right=0.96, hspace=0.22, wspace=0.18)
    return save(fig, "figures/main/F3_specialization")
