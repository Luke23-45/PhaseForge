"""A6 — steps-to-completion ECDFs on solvable tasks (Lift, Can, Square)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import ecdf, save

SOLVABLE_TASKS = ("Lift", "Can", "Square")
METHODS_TO_COMPARE = (
    ("phaseforge", "-", 2.0),
    ("bc", "--", 1.6),
    ("warmstart_moe", ":", 1.6),
    ("plain_encoder_phase_bootstrap", "-.", 1.6),
)


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    with paper_style():
        fig, axes = plt.subplots(
            1, len(SOLVABLE_TASKS), figsize=(7.0, 2.6), squeeze=True, sharey=True
        )
        for col, task in enumerate(SOLVABLE_TASKS):
            ax = axes[col]
            for method, linestyle, linewidth in METHODS_TO_COMPARE:
                if method not in registry.matrix_method_names():
                    continue
                success_steps = []
                for seed in registry.seeds("final"):
                    for ep in dataset.episodes.get((task, method, seed), []):
                        if ep.valid and ep.success and ep.steps > 0:
                            success_steps.append(ep.steps)
                if success_steps:
                    color = method_color(method)
                    med = float(np.median(success_steps))
                    ecdf(
                        ax,
                        success_steps,
                        color=color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        label=f"{registry.display_name(method)} (med: {med:.0f})",
                    )

            ax.set_title(task, fontsize=9.5, fontweight="bold", pad=6)
            ax.set_xlabel("Completion Steps", fontsize=8.5)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, linestyle=":", alpha=0.3)
            if col == 0:
                ax.set_ylabel("Empirical CDF", fontsize=8.5)

        # Unified top legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=False,
            fontsize=7.5,
        )
        fig.subplots_adjust(top=0.82, bottom=0.18, left=0.10, right=0.96, wspace=0.18)
    return save(fig, "figures/appendix/A6_steps_ecdf")
