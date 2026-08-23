"""F5 — initial routing distribution across router initializations.

Plan adjustment (recorded in the module docstring and the final report): the
trainer persists expert frequencies only at the bootstrap instant
(``t0_top1_expert_frequencies``), not per-epoch phase×expert matrices, so the
end-of-training heatmap is not producible from existing artifacts. This figure
shows the *initial* top-1 routing distribution (methods × experts) — the H1
visual: centroid initialization starts structured, random starts flat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from studies.analysis.common import registry
from studies.analysis.common.style import paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import heatmap, save

ROWS = (
    "phaseforge",
    "pf_spherical_kmeans",
    "pf_kmeans",
    "pf_phase_head",
    "pf_random_random",
    "pf_centroid_random",
)


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    rows, matrices = [], []
    first_seed = registry.seeds("final")[0]
    for name in ROWS:
        key = (None if name.startswith("pf_") else "Lift", name, first_seed, 2)
        init = dataset.init_routing.get(key)
        if init is None or not init.t0_top1_expert_frequencies:
            continue
        rows.append(registry.display_name(name))
        matrices.append(np.asarray(init.t0_top1_expert_frequencies, dtype=float))
    if not matrices:
        raise ValueError("No init_routing records found for the router-init family")
    width = max(len(m) for m in matrices)
    matrix = np.full((len(matrices), width), np.nan)
    for i, m in enumerate(matrices):
        matrix[i, : len(m)] = m
    with paper_style():
        fig, ax = plt.subplots(figsize=(4.2, 0.5 * len(rows) + 1.0))
        im = heatmap(
            ax,
            matrix,
            rows,
            [f"e{i}" for i in range(width)],
            cmap="magma",
            vmin=0.0,
            annotate=True,
            fmt="{:.2f}",
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="top-1 share at t=0")
        ax.set_title("Initial routing distribution (t=0)", fontsize=10)
        fig.tight_layout()
    return save(fig, "figures/main/F5_initial_routing")
