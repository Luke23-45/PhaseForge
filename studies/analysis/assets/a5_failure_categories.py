"""A5 — episode outcome / failure-category breakdown per method × task."""

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
    "invalid": OKABE_ITO["grey"],
}


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = registry.tasks()
    labels, share_success, share_timeout, share_invalid = [], [], [], []
    for method in registry.matrix_method_names():
        for task in tasks:
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
                continue
            labels.append(f"{registry.display_name(method)}\n{task}")
            share_success.append(cats.get("success", 0) / total)
            share_timeout.append(cats.get("task_timeout", 0) / total)
            share_other = sum(v for k, v in cats.items() if k not in ("success", "task_timeout"))
            share_invalid.append((cats.get("invalid", 0) + share_other) / total)
    with paper_style():
        fig, ax = plt.subplots(figsize=(7.0, 0.28 * len(labels) + 1.2))
        stacked_bars(
            ax,
            labels,
            {
                "success": share_success,
                "task_timeout": share_timeout,
                "other/invalid": share_invalid,
            },
            colors=CATEGORY_COLORS | {"other/invalid": OKABE_ITO["grey"]},
        )
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
        fig.tight_layout()
    return save(fig, "figures/appendix/A5_failure_categories")
