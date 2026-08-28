"""A5 — episode outcome / failure-category breakdown per method across all 5 tasks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import OKABE_ITO, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import save, stacked_bars

CATEGORY_COLORS = {
    "success": OKABE_ITO["green"],
    "task_timeout": OKABE_ITO["vermillion"],
    "other/invalid": OKABE_ITO["grey"],
}


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = registry.tasks()
    methods = list(registry.matrix_method_names())
    method_labels = [registry.display_name(m) for m in methods]

    with paper_style():
        fig, axes = plt.subplots(1, len(tasks), figsize=(8.5, 3.4))

        for col, task in enumerate(tasks):
            ax = axes[col]
            share_success, share_timeout, share_invalid = [], [], []

            for method in methods:
                cats: Counter[str] = Counter()
                total = 0
                for seed in registry.seeds("final"):
                    key = (task, method, seed)
                    if key not in dataset.episodes:
                        continue
                    for ep in dataset.episodes[key]:
                        total += 1
                        if not ep.valid:
                            cats["invalid"] += 1
                        elif ep.success:
                            cats["success"] += 1
                        else:
                            reason = ep.termination_reason or "other"
                            cats[reason if reason != "success" else "success"] += 1

                if total == 0:
                    share_success.append(0.0)
                    share_timeout.append(0.0)
                    share_invalid.append(0.0)
                    continue

                share_success.append(cats.get("success", 0) / total)
                share_timeout.append(cats.get("task_timeout", 0) / total)
                other_cnt = sum(v for k, v in cats.items() if k not in ("success", "task_timeout"))
                share_invalid.append((cats.get("invalid", 0) + other_cnt) / total)

            stacked_bars(
                ax,
                method_labels if col == 0 else ["" for _ in method_labels],
                {
                    "success": share_success,
                    "task_timeout": share_timeout,
                    "other/invalid": share_invalid,
                },
                colors=CATEGORY_COLORS,
            )
            ax.set_title(task, fontsize=9.5, fontweight="bold", pad=6)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_xticklabels(["0%", "50%", "100%"], fontsize=7.5)
            if col == 0:
                ax.set_yticklabels(method_labels, fontsize=8.0)
            else:
                ax.set_yticklabels([])
            if col == 2:
                ax.set_xlabel("Episode Outcome Share", fontsize=8.5)

        # Single top legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            ["Success", "Task Timeout", "Other / Invalid"],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
        fig.subplots_adjust(top=0.86, bottom=0.14, left=0.25, right=0.97, wspace=0.15)
    return save(fig, "figures/appendix/A5_failure_categories")
