"""T2 — causal mechanism controls on Lift (H1–H4 in one table)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table
from studies.analysis.stats.intervals import mean

CONTROLS = (
    ("phaseforge", "—", "proposed (centroid + phase-pretrain + partial-warm)"),
    ("phase_pretrain_random_router", "H1", "random router (vs centroid)"),
    ("plain_encoder_phase_bootstrap", "H2", "BC encoder (vs phase-supervised)"),
    ("pf_spherical_kmeans", "H3", "generic spherical clustering"),
    ("pf_kmeans", "—", "generic Euclidean clustering"),
    ("pf_phase_head", "H4", "discriminative phase-head directions"),
)


def _sr_and_nmi(dataset: AnalysisDataset, name: str) -> tuple[float | None, float | None, float]:
    rates, nmis = [], []
    seeds = sorted(set(list(registry.seeds("ablation")) + list(registry.seeds("final"))))
    for seed in seeds:
        ev = dataset.evals.get((None, name, seed)) or dataset.evals.get(("Lift", name, seed))
        if ev is not None:
            rates.append(ev.success_rate)
        curve = dataset.curves.get((None, name, seed, 2)) or dataset.curves.get(("Lift", name, seed, 2))
        if curve is not None:
            nmi = curve.last("nmi")
            if nmi is not None:
                nmis.append(nmi)
    sr = mean(rates) if rates else None
    nmi = mean(nmis) if nmis else None
    t0 = None
    for seed in seeds:
        init = dataset.init_routing.get((None, name, seed, 2)) or dataset.init_routing.get(("Lift", name, seed, 2))
        if init is not None and init.t0_nmi is not None:
            t0 = init.t0_nmi
            break
    return sr, nmi, t0 if t0 is not None else float("nan")


def generate(dataset: AnalysisDataset) -> list[Path]:
    rows = []
    pf_sr, _, _ = _sr_and_nmi(dataset, "phaseforge")
    for name, hypothesis, contrast in CONTROLS:
        sr, nmi, t0 = _sr_and_nmi(dataset, name)
        delta = (sr - pf_sr) if (sr is not None and pf_sr is not None) else None
        delta_str = f"{delta:+.2f}" if (delta is not None and name != "phaseforge") else "—"
        display = registry.display_name(name)
        if name == "phaseforge":
            display = r"\textbf{PhaseForge}"
        rows.append(
            [
                display,
                hypothesis,
                contrast,
                f"{sr:.2f}" if sr is not None else "--",
                delta_str,
                f"{t0:.3f}" if t0 == t0 else "--",
                f"{nmi:.3f}" if nmi is not None else "--",
            ]
        )
    table = Table(
        headers=["Method", "H", "Contrast", "SR (Lift)", "Δ vs PF", "NMI t=0", "NMI final"],
        rows=rows,
        caption="Causal mechanism controls on Lift (isolating hypotheses H1–H4; SR mean over seeds; "
        "Δ relative to PhaseForge).",
        notes=(
            "All cells share the R50-matched partial warm-start expert initialization; "
            "each control isolates exactly its declared factor.",
        ),
    )
    return save_table(table, "tables/T2_causal_controls")
