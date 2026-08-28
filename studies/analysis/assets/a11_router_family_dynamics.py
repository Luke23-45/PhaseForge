"""A11 — router-init family dynamics on Lift (NMI / entropy / switch rate / collapse)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import plot_seed_trajectories, save

FAMILY = (
    ("phaseforge", "PhaseForge (Centroid)"),
    ("pf_spherical_kmeans", "Spherical K-Means"),
    ("pf_kmeans", "Euclidean K-Means"),
    ("pf_phase_head", "Phase Head"),
    ("pf_random_random", "Random Router (H1)"),
)
FIELDS = (
    ("nmi", "Phase–Expert NMI"),
    ("routing_entropy", "Routing Entropy"),
    ("switch_rate", "Switch Rate"),
    ("top1_collapse", "Top-1 Collapse"),
)


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    with paper_style():
        fig, axes = plt.subplots(
            len(FIELDS), 1, figsize=(5.4, 6.8), squeeze=True, sharex=True
        )
        for row, (field, ylabel) in enumerate(FIELDS):
            ax = axes[row]
            for name, display in FAMILY:
                per_seed = []
                for seed in sorted(set(list(registry.seeds("ablation")) + list(registry.seeds("final")))):
                    curve = dataset.curves.get((None, name, seed, 2)) or dataset.curves.get(("Lift", name, seed, 2))
                    if curve is not None:
                        series = curve.series(field)
                        if series:
                            per_seed.append(series)
                if not per_seed:
                    continue
                plot_seed_trajectories(
                    ax,
                    per_seed,
                    method_color(name),
                    label=display if row == 0 else None,
                    show_ribbon=True,
                )
            ax.set_ylabel(ylabel, fontsize=8.5)
            ax.grid(True, linestyle=":", alpha=0.3)
            if row == len(FIELDS) - 1:
                ax.set_xlabel("Stage-2 Epoch", fontsize=8.5)

        axes[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.32),
            ncol=3,
            frameon=False,
            fontsize=7.5,
        )
        fig.subplots_adjust(top=0.90, bottom=0.08, left=0.15, right=0.96, hspace=0.22)
    return save(fig, "figures/appendix/A11_router_family")
