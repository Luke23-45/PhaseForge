"""A3 — training curves, all methods × tasks (val action loss + loss decomposition)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import plot_seed_trajectories, save

PANELS = (
    ("val_loss_action", "val action loss"),
    ("train_loss_balance", "balance loss"),
    ("train_loss_teacher_kl", "teacher KL"),
    ("train_lr", "learning rate"),
)


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = registry.tasks()
    methods = registry.matrix_method_names()
    with paper_style():
        fig, axes = plt.subplots(
            len(tasks), len(PANELS), figsize=(11.0, 1.9 * len(tasks)), squeeze=False
        )
        for row, task in enumerate(tasks):
            for col, (field, title) in enumerate(PANELS):
                ax = axes[row][col]
                for method in methods:
                    per_seed = []
                    for seed in registry.seeds("final"):
                        key = (task, method, seed, _final_stage(dataset, method))
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
                        label=registry.display_name(method) if row == 0 and col == 0 else None,
                    )
                ax.set_title(f"{task} — {title}" if row == 0 else title, fontsize=8)
                if col == 0:
                    ax.set_ylabel(task if row == 0 else "")
        fig.tight_layout()
    return save(fig, "figures/appendix/A3_training_curves")


def _final_stage(dataset: AnalysisDataset, method: str) -> int:
    for m in registry.methods("final"):
        if m.name == method:
            return m.stages[-1]
    return 1
