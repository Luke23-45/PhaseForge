"""A8 — expert load balance score trajectories over Stage-2 training."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import OKABE_ITO, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import plot_seed_trajectories, save

FIELDS = (("top1_balance", "Top-1 Balance Score"), ("topk_balance", "Top-k Balance Score"))
TASKS = ("Lift", "Can")


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [t for t in TASKS if t in registry.tasks()]
    with paper_style():
        fig, axes = plt.subplots(
            len(FIELDS), len(tasks), figsize=(6.2, 3.8), squeeze=False, sharex=True, sharey=True
        )
        for col, task in enumerate(tasks):
            for row, (field, ylabel) in enumerate(FIELDS):
                ax = axes[row][col]
                per_seed = []
                for seed in registry.seeds("final"):
                    key = (task, "phaseforge", seed, 2)
                    if key in dataset.curves:
                        series = dataset.curves[key].series(field)
                        if series:
                            per_seed.append(series)
                if per_seed:
                    plot_seed_trajectories(
                        ax,
                        per_seed,
                        OKABE_ITO["vermillion"],
                        label="PhaseForge (mean ± range)",
                        xlabel="Stage-2 Epoch" if row == len(FIELDS) - 1 else "",
                        show_ribbon=True,
                    )

                ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8, label="Ideal (1.0)")
                ax.set_ylim(0.82, 1.02)
                ax.grid(True, linestyle=":", alpha=0.3)
                if col == 0:
                    ax.set_ylabel(ylabel, fontsize=9)
                if row == len(FIELDS) - 1:
                    ax.set_xlabel("Stage-2 Epoch", fontsize=9)
                if row == 0:
                    ax.set_title(task, fontsize=9.5, fontweight="bold", pad=6)

        axes[0][0].legend(frameon=False, fontsize=7.5, loc="lower right")
        fig.tight_layout()
    return save(fig, "figures/appendix/A8_balance")
