"""A6 — steps-to-outcome ECDFs (success and timeout episodes separated)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import ecdf, save


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = registry.tasks()
    with paper_style():
        fig, axes = plt.subplots(
            1, len(tasks), figsize=(1.9 * len(tasks) + 1.2, 2.4), squeeze=False, sharey=True
        )
        for col, task in enumerate(tasks):
            ax = axes[0][col]
            for method in ("phaseforge", "bc", "warmstart_moe"):
                success_steps, timeout_steps = [], []
                for seed in registry.seeds("final"):
                    for ep in dataset.episodes.get((task, method, seed), []):
                        if not ep.valid or ep.steps <= 0:
                            continue
                        (success_steps if ep.success else timeout_steps).append(ep.steps)
                color = method_color(method)
                if success_steps:
                    ecdf(
                        ax,
                        success_steps,
                        color=color,
                        label=f"{registry.display_name(method)} (succ.)",
                    )
                if timeout_steps:
                    ecdf(ax, timeout_steps, color=color, label=None)
                    ax.plot([], [])  # keep ordering stable
            ax.set_title(task, fontsize=9)
            ax.set_xlabel("steps")
            if col == 0:
                ax.set_ylabel("ECDF")
        axes[0][0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
    return save(fig, "figures/appendix/A6_steps_ecdf")
