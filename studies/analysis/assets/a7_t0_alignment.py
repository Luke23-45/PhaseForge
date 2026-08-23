"""A7 — t=0 routing alignment across router initializations (H1 initial condition)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table

FAMILY = (
    "phaseforge",
    "pf_spherical_kmeans",
    "pf_kmeans",
    "pf_phase_head",
    "pf_random_random",
    "pf_centroid_random",
    "phase_pretrain_random_router",
)


def generate(dataset: AnalysisDataset) -> list[Path]:
    rows = []
    first_seed = registry.seeds("ablation")[0]
    for name in FAMILY:
        task = "Lift" if name == "phaseforge" else None
        init = dataset.init_routing.get((task, name, first_seed, 2))
        if init is None:
            continue
        freqs = init.t0_top1_expert_frequencies
        dead = init.t0_dead_expert_count
        top_share = max(freqs) if freqs else float("nan")
        rows.append(
            [
                registry.display_name(name),
                f"{init.t0_nmi:.3f}",
                f"{init.t0_routing_entropy:.3f}",
                f"{init.t0_normalized_routing_entropy:.3f}"
                if init.t0_normalized_routing_entropy is not None
                else "--",
                f"{top_share:.3f}" if top_share == top_share else "--",
                str(dead) if dead is not None else "--",
                f"{init.t0_phase_head_accuracy:.3f}"
                if init.t0_phase_head_accuracy is not None
                else "--",
            ]
        )
    table = Table(
        headers=[
            "Router init",
            "t=0 NMI",
            "Entropy",
            "Norm. entropy",
            "Max top-1 share",
            "Dead experts",
            "Phase-head acc.",
        ],
        rows=rows,
        caption="Bootstrap-instant routing diagnostics per router initialization "
        "(metadata/init\\_routing.json; seed shown in provenance, A10).",
        notes=(
            "Centroid initialization starts phase-aligned (high t=0 NMI); random "
            "initializations start near-uniform.",
        ),
    )
    return save_table(table, "tables/A7_t0_alignment")
