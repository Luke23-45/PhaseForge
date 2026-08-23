"""F3 — Stage-2 specialization dynamics: NMI, routing entropy, switch rate.

PhaseForge vs the H1/H2 controls and the scratch floor on Lift (plus Can when
present); thin lines are seeds, the bold line is the seed mean; t=0 is the
bootstrap instant recorded in init_routing.json.
"""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import plot_seed_trajectories, save

METHODS = (
    "phaseforge",
    "phase_pretrain_random_router",
    "plain_encoder_phase_bootstrap",
    "scratch_moe",
)
FIELDS = (
    ("nmi", "phase–expert NMI"),
    ("routing_entropy", "routing entropy"),
    ("switch_rate", "switch rate"),
)


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [t for t in ("Lift", "Can") if t in registry.tasks()]
    method_names = [m for m in METHODS if m in registry.matrix_method_names()]
    with paper_style():
        fig, axes = plt.subplots(
            len(FIELDS),
            len(tasks),
            figsize=(7.0, 2.2 * len(FIELDS)),
            squeeze=False,
            sharex="col",
        )
        for col, task in enumerate(tasks):
            for row, (field, ylabel) in enumerate(FIELDS):
                ax = axes[row][col]
                for method in method_names:
                    per_seed = []
                    for seed in registry.seeds("final"):
                        key = (task, method, seed, 2)
                        if key in dataset.curves:
                            series = dataset.curves[key].series(field)
                            if series:
                                per_seed.append(series)
                    if not per_seed:
                        continue
                    init = dataset.init_routing.get((task, method, registry.seeds("final")[0], 2))
                    if init is not None and field == "nmi" and init.t0_nmi is not None:
                        pass  # t0 marker drawn after trajectories
                    plot_seed_trajectories(
                        ax,
                        per_seed,
                        method_color(method),
                        label=registry.display_name(method) if row == 0 else None,
                        ylabel=ylabel if col == 0 else "",
                        xlabel="Stage-2 epoch" if row == len(FIELDS) - 1 else "",
                    )
                if row == 0 and col == 0:
                    ax.legend(frameon=False, loc="best")
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row == len(FIELDS) - 1:
                    ax.set_xlabel("Stage-2 epoch")
                if row == 0:
                    ax.set_title(task)
        fig.tight_layout()
    return save(fig, "figures/main/F3_specialization")
