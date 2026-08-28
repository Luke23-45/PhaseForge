"""A3 — training curves across all 5 tasks (validation action loss and expert balance loss)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import plot_seed_trajectories, save

PANELS = (
    ("val_loss_action", "Validation Action MSE"),
    ("train_loss_balance", "Expert Balance Loss"),
)
METHODS_TO_PLOT = (
    "phaseforge",
    "bc",
    "bc_large",
    "scratch_moe",
    "warmstart_moe",
    "plain_encoder_phase_bootstrap",
)


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = registry.tasks()
    with paper_style():
        fig, axes = plt.subplots(
            len(tasks), len(PANELS), figsize=(7.2, 7.8), squeeze=False, sharex=True
        )
        for row, task in enumerate(tasks):
            for col, (field, title) in enumerate(PANELS):
                ax = axes[row][col]
                for method in METHODS_TO_PLOT:
                    if method not in registry.matrix_method_names():
                        continue
                    per_seed = []
                    for seed in registry.seeds("final"):
                        key = (task, method, seed, _final_stage(method))
                        if key in dataset.curves:
                            series = dataset.curves[key].series(field)
                            if series:
                                per_seed.append(series)
                    if not per_seed:
                        continue
                    plot_seed_trajectories(
                        ax,
                        per_seed,
                        method_color(method),
                        label=registry.display_name(method) if (row == 0 and col == 0) else None,
                        show_ribbon=True,
                    )

                ax.grid(True, linestyle=":", alpha=0.3)
                if row == 0:
                    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
                if col == 0:
                    ax.set_ylabel(f"{task}\nMSE", fontsize=8.5, fontweight="bold")
                if row == len(tasks) - 1:
                    ax.set_xlabel("Epoch", fontsize=8.5)

        # Single clean outside legend
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(
            h,
            l,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
        fig.subplots_adjust(top=0.88, bottom=0.06, left=0.12, right=0.96, hspace=0.22, wspace=0.20)
    return save(fig, "figures/appendix/A3_training_curves")


def _final_stage(method: str) -> int:
    for m in registry.methods("final"):
        if m.name == method:
            return m.stages[-1]
    return 1
