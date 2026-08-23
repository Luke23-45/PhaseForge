"""T1 — five-task success matrix (the result table).

Mean SR over seeds with the pooled Wilson interval; macro-average column
marked secondary; PhaseForge row bolded via LaTeX markup; paired Δ vs BC
reported in a footer band (full tests in A15).
"""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table
from studies.analysis.stats.intervals import mean, sample_std, seed_mean_and_wilson


def generate(dataset: AnalysisDataset) -> list[Path]:
    tasks = list(registry.tasks())
    headers = ["Method"] + tasks + ["Macro-avg"]
    rows: list[list[str]] = []
    for method in registry.matrix_method_names():
        cells: list[str] = []
        all_rates: list[float] = []
        for task in tasks:
            per_seed_successes, per_seed_rates = [], []
            for seed in registry.seeds("final"):
                key = (task, method, seed)
                if key not in dataset.evals:
                    cells.append("--")
                    continue
                ev = dataset.evals[key]
                per_seed_successes.append(ev.successes)
                per_seed_rates.append(ev.success_rate)
            if not per_seed_rates:
                continue
            p, lo, hi = seed_mean_and_wilson(
                per_seed_successes,
                dataset.evals[(task, method, registry.seeds("final")[0])].valid_episodes,
            )
            all_rates.extend(per_seed_rates)
            cells.append(f"{p:.2f} [{lo:.2f}, {hi:.2f}]")
        macro = f"{mean(all_rates):.2f} ± {sample_std(all_rates):.2f}" if all_rates else "--"
        display = registry.display_name(method)
        if method == "phaseforge":
            display = r"\textbf{PhaseForge}"
        rows.append([display] + cells + [macro])

    table = Table(
        headers=headers,
        rows=rows,
        caption="Rollout success rate per task (mean over seeds; brackets: pooled Wilson 95\\% "
        "CI; macro-average ± sample std across task-seeds, secondary aggregate).",
        notes=(
            "Teacher-Forced is a privileged-training diagnostic, not a deployable "
            "method. Paired Δ vs BC with Holm-adjusted tests: asset A15.",
        ),
    )
    return save_table(table, "tables/T1_success_matrix")
