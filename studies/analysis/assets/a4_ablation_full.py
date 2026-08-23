"""A4 — full ablation table (every ablation-namespace cell)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table


def _cell_summary(dataset: AnalysisDataset, name: str) -> list[str]:
    rates, nmis, collapses = [], [], []
    for seed in registry.seeds("ablation"):
        key = (None, name, seed)
        if key not in dataset.evals:
            continue
        rates.append(dataset.evals[key].success_rate)
        curve = dataset.curves.get((None, name, seed, 2))
        if curve is not None:
            nmi = curve.last("nmi")
            col = curve.last("top1_collapse")
            if nmi is not None:
                nmis.append(nmi)
            if col is not None:
                collapses.append(col)
    sr = f"{sum(rates) / len(rates):.2f}" if rates else "--"
    seeds_str = ", ".join(f"{r:.2f}" for r in rates) if rates else "--"
    nmi = f"{sum(nmis) / len(nmis):.3f}" if nmis else "--"
    col = f"{sum(collapses) / len(collapses):.2f}" if collapses else "--"
    return [sr, seeds_str, nmi, col]


def generate(dataset: AnalysisDataset) -> list[Path]:
    rows = []
    for method in registry.methods("ablation"):
        name = method.name
        if (
            name in registry.matrix_method_names()
            and name in {m.name for m in registry.methods("final")}
            and method.experiment_id
            and method.experiment_id.startswith("EXP-1")
        ):
            role = "matrix replica"
        elif name in registry.ablation_method_names():
            role = "ablation"
        else:
            role = "matrix replica"
        rows.append(
            [method.experiment_id or "—", registry.display_name(name), role]
            + _cell_summary(dataset, name)
        )
    table = Table(
        headers=[
            "EXP",
            "Method",
            "Role",
            "SR (mean)",
            "SR (per seed)",
            "NMI final",
            "Collapse final",
        ],
        rows=rows,
        caption="Full ablation suite (Lift namespace; every cell, seeds "
        f"{list(registry.seeds('ablation'))}).",
        notes=("pf\\_drop50 is the canonical PhaseForge configuration (R50).",),
    )
    return save_table(table, "tables/A4_ablation_full")
