"""A14 — phase-depth analysis: distribution of deepest phase reached (max_phase)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import save


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = registry.tasks()
    methods = [
        m for m in ("phaseforge", "bc", "warmstart_moe", "plain_encoder_phase_bootstrap")
        if m in registry.matrix_method_names()
    ]
    with paper_style():
        fig, axes = plt.subplots(
            1, len(tasks), figsize=(7.2, 2.5), squeeze=True, sharey=True
        )
        for col, task in enumerate(tasks):
            ax = axes[col]
            depths_by_method: dict[str, Counter[int]] = {}
            for method in methods:
                counter: Counter[int] = Counter()
                for seed in registry.seeds("final"):
                    for ep in dataset.episodes.get((task, method, seed), []):
                        if ep.max_phase is not None:
                            counter[ep.max_phase] += 1
                if counter:
                    depths_by_method[method] = counter
            if not depths_by_method:
                ax.set_visible(False)
                continue
            max_depth = 5  # Standard benchmark phases 0..5 (P=6)
            width = 0.8 / len(depths_by_method)
            for i, (method, counter) in enumerate(depths_by_method.items()):
                total = sum(counter.values())
                xs = np.arange(max_depth + 1) + (i - len(depths_by_method) / 2 + 0.5) * width
                ax.bar(
                    xs,
                    [counter.get(d, 0) / total for d in range(max_depth + 1)],
                    width=width,
                    color=method_color(method),
                    label=registry.display_name(method) if col == 0 else None,
                    edgecolor="white",
                    linewidth=0.4,
                )
            ax.axvline(5.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_xlabel("Phase Reached", fontsize=8.0)
            ax.set_xticks(range(max_depth + 1))
            ax.set_xticklabels([f"P{d}" for d in range(max_depth + 1)], fontsize=7.0)
            ax.set_title(task, fontsize=9.0, fontweight="bold", pad=5)
            ax.grid(axis="y", linestyle=":", alpha=0.3)
            if col == 0:
                ax.set_ylabel("Share of Episodes", fontsize=8.5)

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
        fig.subplots_adjust(top=0.84, bottom=0.18, left=0.08, right=0.98, wspace=0.15)
    return save(fig, "figures/appendix/A14_phase_depth")
