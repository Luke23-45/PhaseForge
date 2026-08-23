"""F2 — per-task paired success deltas (PhaseForge vs key baselines/controls).

Forest panels, one per comparison method; rows are tasks; the point is the
seed-mean paired delta computed per-episode on identical reset cases, the
interval spans the per-seed deltas (seed points drawn individually).
"""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import OKABE_ITO, method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import forest, save
from studies.analysis.stats.paired import pair_episodes

COMPARISONS = (
    "bc",
    "warmstart_moe",
    "phase_pretrain_random_router",
    "plain_encoder_phase_bootstrap",
)
PANEL_TITLES = {
    "bc": "PhaseForge − BC",
    "warmstart_moe": "PhaseForge − Warm-Start MoE",
    "phase_pretrain_random_router": "PhaseForge − PP Random-Router (H1)",
    "plain_encoder_phase_bootstrap": "PhaseForge − PE Phase-Bootstrap (H2)",
}


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = registry.tasks()
    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), squeeze=False, sharex=True)
        for ax, comparator in zip(axes.flat, COMPARISONS):
            labels, means, lows, highs, seed_dots = [], [], [], [], []
            for task in tasks:
                seed_deltas = []
                for seed in registry.seeds("final"):
                    key_a = (task, "phaseforge", seed)
                    key_b = (task, comparator, seed)
                    if key_a not in dataset.episodes or key_b not in dataset.episodes:
                        continue
                    bank_a = dataset.evals[key_a].reset_bank
                    bank_b = dataset.evals[key_b].reset_bank
                    if bank_a != bank_b:
                        continue  # pairing invalid across different banks
                    outcome = pair_episodes(
                        task,
                        seed,
                        dataset.episodes[key_a],
                        dataset.episodes[key_b],
                        bank_a=bank_a,
                        bank_b=bank_b,
                    )
                    seed_deltas.append(outcome.delta)
                if not seed_deltas:
                    continue
                labels.append(task)
                mean_delta = sum(seed_deltas) / len(seed_deltas)
                means.append(mean_delta)
                lows.append(min(seed_deltas))
                highs.append(max(seed_deltas))
                seed_dots.extend((task, d) for d in seed_deltas)
            color = method_color(comparator)
            forest(
                ax,
                labels,
                means,
                lows,
                highs,
                colors=[color] * len(labels),
                xlabel="Δ success rate",
            )
            # individual seed deltas as open markers beside the mean point
            for i, label in enumerate(labels):
                for task_name, delta in seed_dots:
                    if task_name == label:
                        ax.scatter(
                            [delta],
                            [i],
                            facecolors="none",
                            edgecolors=color,
                            s=18,
                            linewidths=0.9,
                            zorder=3,
                        )
            ax.set_title(PANEL_TITLES[comparator], fontsize=10)
            ax.axvline(0.0, color=OKABE_ITO["grey"], linewidth=0.8, linestyle="--")
        fig.tight_layout()
    return save(fig, "figures/main/F2_paired_deltas")
