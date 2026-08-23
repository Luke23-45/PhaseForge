"""A8 — expert balance-score trajectories over Stage-2.

Plan adjustment: per-expert utilization time series are not persisted (only
balance/collapse scores are), so the stacked-area view is replaced by the
top1/topk balance-score trajectories — same diagnostic family, producible
from the audited schema.
"""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import OKABE_ITO, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import plot_seed_trajectories, save

FIELDS = (("top1_balance", "top-1 balance"), ("topk_balance", "top-k balance"))
TASKS = ("Lift", "Can")


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [t for t in TASKS if t in registry.tasks()]
    with paper_style():
        fig, axes = plt.subplots(
            len(FIELDS), len(tasks), figsize=(2.6 * len(tasks) + 0.8, 2.4), squeeze=False
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
                        xlabel="Stage-2 epoch" if row == len(FIELDS) - 1 else "",
                    )
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row == len(FIELDS) - 1:
                    ax.set_xlabel("Stage-2 epoch")
                if row == 0:
                    ax.set_title(task, fontsize=9)
        fig.tight_layout()
    return save(fig, "figures/appendix/A8_balance")
