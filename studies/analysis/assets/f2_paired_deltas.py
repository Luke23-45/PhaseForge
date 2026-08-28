"""F2 — per-task paired success deltas (PhaseForge vs key baselines/controls).

1x4 horizontal strip forest plot; rows are tasks; point is the paired seed-mean delta
on identical reset cases; intervals span min-max seed deltas with distinct caps;
individual seed deltas are vertically jittered (n=3) to prevent overplotting.
"""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import registry
from studies.analysis.common.style import method_color, paper_style
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
    "bc": "vs. BC",
    "warmstart_moe": "vs. Warm-Start",
    "phase_pretrain_random_router": "vs. Rand-Router (H1)",
    "plain_encoder_phase_bootstrap": "vs. Phase-Boot (H2)",
}


def generate(dataset: AnalysisDataset) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = list(reversed(registry.tasks()))  # Lift at top
    with paper_style():
        fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.5), squeeze=True, sharey=True, sharex=True)
        for col_idx, (ax, comparator) in enumerate(zip(axes, COMPARISONS)):
            labels, means, lows, highs, seed_points = [], [], [], [], []
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
                seed_points.append(seed_deltas)

            color = method_color(comparator)
            forest(
                ax,
                labels,
                means,
                lows,
                highs,
                colors=[color] * len(labels),
                seed_points=seed_points,
                xlabel="Δ success rate" if col_idx == 0 or col_idx == 2 else "",
                show_zero=True,
                capsize=3.0,
            )
            ax.set_title(PANEL_TITLES[comparator], fontsize=9.0, fontweight="bold", pad=6)
            ax.set_xlim(-0.45, 0.45)

            # Add subtle grid and reference lines
            ax.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
            ax.grid(axis="x", linestyle=":", alpha=0.3)

        fig.subplots_adjust(top=0.86, bottom=0.18, left=0.12, right=0.98, wspace=0.18)
    return save(fig, "figures/main/F2_paired_deltas")
