"""A8 — expert load balance score trajectories over Stage-2 training."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import OKABE_ITO, method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import plot_seed_trajectories, save

FIELDS = (("top1_balance", "Top-1 Balance Score"), ("topk_balance", "Top-k Balance Score"))
TASKS = ("Lift", "Can")


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [t for t in TASKS if t in registry.tasks()]
    with paper_style():
        fig, axes = plt.subplots(
            len(FIELDS), len(tasks), figsize=(6.2, 3.8), squeeze=False, sharex=True, sharey=True
        )
        for col, task in enumerate(tasks):
            for row, (field, ylabel) in enumerate(FIELDS):
                ax = axes[row][col]
                moe_methods = [
                    ("precision_residual_phaseforge", "PhaseForge"),
                    ("final_aligned_softmax_top1", "Softmax Top-1"),
                    ("precision_residual_phase_random_router", "Phase-Random"),
                    ("precision_residual_scratch_moe", "Scratch MoE"),
                ]
                for method_id, display in moe_methods:
                    color = method_color(method_id)
                    per_seed = []
                    for seed in registry.seeds("final"):
                        for m_cand in (method_id, "phaseforge" if method_id == "precision_residual_phaseforge" else method_id):
                            key = (task, m_cand, seed, 2)
                            if key in dataset.curves:
                                series = dataset.curves[key].series(field)
                                if series:
                                    per_seed.append(series)
                                break
                    if per_seed:
                        plot_seed_trajectories(
                            ax,
                            per_seed,
                            color,
                            label=display if (row == 0 and col == 0) else None,
                            xlabel="Stage-2 Epoch" if row == len(FIELDS) - 1 else "",
                            show_ribbon=True,
                        )

                ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8, label="Ideal (1.0)")
                ax.set_ylim(0.82, 1.02)
                ax.grid(True, linestyle=":", alpha=0.3)
                if col == 0:
                    ax.set_ylabel(ylabel, fontsize=9)
                if row == len(FIELDS) - 1:
                    ax.set_xlabel("Stage-2 Epoch", fontsize=9)
                if row == 0:
                    ax.set_title(task, fontsize=9.5, fontweight="bold", pad=6)

        # Top legend to avoid occluding trajectory curves
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=4,
            frameon=False,
            fontsize=8.0,
        )
        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.14, right=0.96, hspace=0.25, wspace=0.18)
    return save(fig, "figures/appendix/A8_balance")
