"""F5 — initial routing distribution and representation alignment across router initializations.

Left panel: Top-1 expert allocation distribution at bootstrap instant (t=0) across 6 router
initialization strategies (experts e0..e5).
Right panel: Bootstrap-instant Phase–Expert NMI and Phase-Head classification accuracy,
demonstrating that centroid initialization starts structured and phase-aligned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from studies.analysis.common import registry
from studies.analysis.common.style import OKABE_ITO, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import heatmap, save

ROWS = (
    ("phaseforge", "PhaseForge (Centroid)"),
    ("pf_spherical_kmeans", "Spherical K-Means"),
    ("pf_kmeans", "Euclidean K-Means"),
    ("pf_phase_head", "Phase Head Directions"),
    ("pf_random_random", "Random Router (H1)"),
    ("pf_centroid_random", "Centroid + Rand Exp"),
)


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    rows, matrices, nmis, accs = [], [], [], []
    first_seed = registry.seeds("final")[0]

    for name, display in ROWS:
        key = (None if name.startswith("pf_") else "Lift", name, first_seed, 2)
        init = dataset.init_routing.get(key)
        if init is None or not init.t0_top1_expert_frequencies:
            continue
        rows.append(display)
        matrices.append(np.asarray(init.t0_top1_expert_frequencies, dtype=float))
        nmis.append(float(init.t0_nmi) if init.t0_nmi is not None else 0.0)
        accs.append(float(init.t0_phase_head_accuracy) if init.t0_phase_head_accuracy is not None else 0.0)

    if not matrices:
        raise ValueError("No init_routing records found for the router-init family")

    width = max(len(m) for m in matrices)
    matrix = np.full((len(matrices), width), np.nan)
    for i, m in enumerate(matrices):
        matrix[i, : len(m)] = m

    with paper_style():
        fig, (ax_heat, ax_bar) = plt.subplots(
            1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.4, 1.0]}
        )

        # Panel A: Allocation Matrix
        im = heatmap(
            ax_heat,
            matrix,
            rows,
            [f"e{i}" for i in range(width)],
            cmap="Blues",
            vmin=0.0,
            vmax=0.6,
            annotate=True,
            fmt="{:.2f}",
        )
        ax_heat.set_title("A. Initial Top-1 Allocation (t=0)", fontsize=9.5, fontweight="bold")
        cbar = fig.colorbar(im, ax=ax_heat, shrink=0.85, pad=0.03)
        cbar.ax.tick_params(labelsize=7.5)
        cbar.set_label("Expert share", fontsize=8)

        # Panel B: Alignment Metrics (NMI & Phase Head Acc)
        y_pos = np.arange(len(rows))
        bar_height = 0.35
        ax_bar.barh(
            y_pos - bar_height / 2,
            nmis,
            height=bar_height,
            color=OKABE_ITO["vermillion"],
            label="t=0 NMI",
            edgecolor="white",
            linewidth=0.5,
        )
        ax_bar.barh(
            y_pos + bar_height / 2,
            accs,
            height=bar_height,
            color=OKABE_ITO["sky"],
            label="Phase-Head Acc",
            edgecolor="white",
            linewidth=0.5,
        )
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([])  # shared row order with left panel
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Alignment Score", fontsize=9)
        ax_bar.set_xlim(0.0, 0.82)
        ax_bar.grid(axis="x", linestyle=":", alpha=0.3)
        ax_bar.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            frameon=False,
            fontsize=7.5,
        )
        ax_bar.set_title("B. Representation Alignment", fontsize=9.5, fontweight="bold", pad=16)

        # Value annotations on bars
        for idx, (n_val, a_val) in enumerate(zip(nmis, accs)):
            ax_bar.text(n_val + 0.015, idx - bar_height / 2, f"{n_val:.3f}", va="center", fontsize=7.5)

        fig.subplots_adjust(top=0.82, bottom=0.16, left=0.25, right=0.96, wspace=0.28)
    return save(fig, "figures/main/F5_initial_routing")
