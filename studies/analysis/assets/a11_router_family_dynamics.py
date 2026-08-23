"""A11 — router-init family dynamics on Lift (entropy / NMI / switch / collapse)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import plot_seed_trajectories, save

FAMILY = ("phaseforge", "pf_spherical_kmeans", "pf_kmeans", "pf_phase_head", "pf_random_random")
FIELDS = (
    ("nmi", "phase–expert NMI"),
    ("routing_entropy", "routing entropy"),
    ("switch_rate", "switch rate"),
    ("top1_collapse", "top-1 collapse"),
)


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    with paper_style():
        fig, axes = plt.subplots(
            len(FIELDS), 1, figsize=(4.4, 1.9 * len(FIELDS)), squeeze=False, sharex=True
        )
        for row, (field, ylabel) in enumerate(FIELDS):
            ax = axes[row][0]
            for name in FAMILY:
                per_seed = []
                for seed in registry.seeds("ablation"):
                    key = (None, name, seed, 2)
                    if key in dataset.curves:
                        series = dataset.curves[key].series(field)
                        if series:
                            per_seed.append(series)
                if not per_seed:
                    continue
                plot_seed_trajectories(
                    ax,
                    per_seed,
                    method_color(name),
                    label=registry.display_name(name) if row == 0 else None,
                )
            ax.set_ylabel(ylabel)
            if row == len(FIELDS) - 1:
                ax.set_xlabel("Stage-2 epoch")
        axes[0][0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
    return save(fig, "figures/appendix/A11_router_family")
