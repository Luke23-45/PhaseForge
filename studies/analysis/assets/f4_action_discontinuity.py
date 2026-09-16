"""F4 — Commanded Action Discontinuity at Expert Switches.

Demonstrates the physical mechanism underlying the decoupling of routing organization
and closed-loop success:
Panel A: Two-task small-multiple layout of per-timestep action jumps ||Δa_t||_2 at
         expert switches vs. non-switch steps across ablation arms (Can and Square).
         Box-and-whisker summaries (median, IQR, 5th-95th percentile whiskers) with
         overlaid per-seed means (n=3) and sample counts.
Panel B: Expert switch rate across ablation arms (including BC as zero-switch reference)
         showing seed-level points, descriptive mean, and observed range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from studies.analysis.common import io as cio
from studies.analysis.common import registry
from studies.analysis.common.config import namespace_root
from studies.analysis.common.style import OKABE_ITO, method_color, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import save

METHODS_MOE = (
    "router_init_topology",
    "router_init_phase",
    "router_init_random",
    "routing_softmax_top1",
)
METHODS_ALL = METHODS_MOE + ("representation_bc",)
TASKS = ("Can", "Square")


def _extract_trace_metrics(dataset: AnalysisDataset):
    """Extract action jumps and switch rates directly from validated trace.jsonl files."""
    data = {
        task: {
            m: {
                "sw_jumps": [],
                "non_jumps": [],
                "seed_sw_means": [],
                "seed_non_means": [],
                "seed_sw_rates": [],
            }
            for m in METHODS_ALL
        }
        for task in TASKS
    }

    seeds = registry.seeds("ablation")
    for task in TASKS:
        for m in METHODS_ALL:
            for s in seeds:
                key = (task, m, s)
                if key not in dataset.eval_runs:
                    continue
                tp = dataset.eval_runs[key].path / "trace.jsonl"
                if not tp.is_file():
                    continue

                cur_ep = None
                prev_act = None
                prev_exp = None
                s_sw_jumps = []
                s_non_jumps = []

                for rec in cio.iter_jsonl(tp):
                    ep = rec.get("episode_id")
                    exp = rec.get("selected_expert")
                    act = rec.get("final_action")
                    if not act:
                        continue
                    act = np.array(act, dtype=float)

                    if ep != cur_ep:
                        cur_ep = ep
                        prev_act = act
                        prev_exp = exp
                        continue

                    if prev_act is not None and len(act) == len(prev_act):
                        jump = float(np.linalg.norm(act - prev_act))
                        if exp != prev_exp:
                            s_sw_jumps.append(jump)
                        else:
                            s_non_jumps.append(jump)

                    prev_act = act
                    prev_exp = exp

                total_steps = len(s_sw_jumps) + len(s_non_jumps)
                sw_rate = len(s_sw_jumps) / total_steps if total_steps > 0 else 0.0

                data[task][m]["seed_sw_rates"].append(sw_rate)
                data[task][m]["sw_jumps"].extend(s_sw_jumps)
                data[task][m]["non_jumps"].extend(s_non_jumps)
                if s_sw_jumps:
                    data[task][m]["seed_sw_means"].append(float(np.mean(s_sw_jumps)))
                if s_non_jumps:
                    data[task][m]["seed_non_means"].append(float(np.mean(s_non_jumps)))

    return data


def generate(dataset: AnalysisDataset) -> list[Path]:
    trace_data = _extract_trace_metrics(dataset)

    with paper_style():
        # 3 subplots: Panel A1 (Can Jumps), Panel A2 (Square Jumps), Panel B (Switch Rates)
        fig = plt.figure(figsize=(7.2, 4.2))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.08, 1.25], wspace=0.46)

        ax_can = fig.add_subplot(gs[0])
        ax_sq = fig.add_subplot(gs[1], sharey=ax_can)
        ax_b = fig.add_subplot(gs[2])

        whisker_rule = (5, 95)  # 5th and 95th percentiles

        # --- PANEL A: Small-multiple boxplots (Can and Square) ---
        for ax, task in [(ax_can, "Can"), (ax_sq, "Square")]:
            n_methods = len(METHODS_MOE)
            positions = np.arange(n_methods)
            width = 0.32

            for i, m in enumerate(METHODS_MOE):
                sw_j = trace_data[task][m]["sw_jumps"]
                non_j = trace_data[task][m]["non_jumps"]
                sw_means = trace_data[task][m]["seed_sw_means"]
                non_means = trace_data[task][m]["seed_non_means"]

                pos_sw = positions[i] + width / 2
                pos_non = positions[i] - width / 2

                # Compute custom percentiles for clean whiskers
                if sw_j:
                    q1, med, q3 = np.percentile(sw_j, [25, 50, 75])
                    w_lo, w_hi = np.percentile(sw_j, whisker_rule)
                    # Draw box
                    ax.fill_between([pos_sw - width*0.4, pos_sw + width*0.4], [q1, q1], [q3, q3],
                                    facecolor=OKABE_ITO["vermillion"], alpha=0.3, edgecolor=OKABE_ITO["vermillion"], lw=1.0)
                    ax.plot([pos_sw - width*0.4, pos_sw + width*0.4], [med, med], color=OKABE_ITO["vermillion"], lw=1.6)
                    # Whiskers
                    ax.plot([pos_sw, pos_sw], [q1, w_lo], color=OKABE_ITO["vermillion"], lw=1.0)
                    ax.plot([pos_sw, pos_sw], [q3, w_hi], color=OKABE_ITO["vermillion"], lw=1.0)
                    ax.plot([pos_sw - width*0.2, pos_sw + width*0.2], [w_lo, w_lo], color=OKABE_ITO["vermillion"], lw=1.0)
                    ax.plot([pos_sw - width*0.2, pos_sw + width*0.2], [w_hi, w_hi], color=OKABE_ITO["vermillion"], lw=1.0)

                    # Overlay per-seed means
                    for sm in sw_means:
                        ax.scatter([pos_sw], [sm], facecolors="none", edgecolors=OKABE_ITO["vermillion"],
                                   s=24, linewidths=1.2, zorder=5)

                if non_j:
                    q1, med, q3 = np.percentile(non_j, [25, 50, 75])
                    w_lo, w_hi = np.percentile(non_j, whisker_rule)
                    # Draw box
                    ax.fill_between([pos_non - width*0.4, pos_non + width*0.4], [q1, q1], [q3, q3],
                                    facecolor=OKABE_ITO["sky"], alpha=0.3, edgecolor=OKABE_ITO["sky"], lw=1.0)
                    ax.plot([pos_non - width*0.4, pos_non + width*0.4], [med, med], color=OKABE_ITO["sky"], lw=1.6)
                    # Whiskers
                    ax.plot([pos_non, pos_non], [q1, w_lo], color=OKABE_ITO["sky"], lw=1.0)
                    ax.plot([pos_non, pos_non], [q3, w_hi], color=OKABE_ITO["sky"], lw=1.0)
                    ax.plot([pos_non - width*0.2, pos_non + width*0.2], [w_lo, w_lo], color=OKABE_ITO["sky"], lw=1.0)
                    ax.plot([pos_non - width*0.2, pos_non + width*0.2], [w_hi, w_hi], color=OKABE_ITO["sky"], lw=1.0)

                    # Overlay per-seed means
                    for nm in non_means:
                        ax.scatter([pos_non], [nm], facecolors="none", edgecolors=OKABE_ITO["sky"],
                                   s=24, linewidths=1.2, zorder=5)

            # Clean x-tick labels with sample counts attached
            xtick_labels = []
            for m in METHODS_MOE:
                short = registry.display_name(m).replace(" Init", "").replace(" (PF)", "")
                n_sw = len(trace_data[task][m]["sw_jumps"])
                xtick_labels.append(f"{short}\n($N={n_sw}$)")

            ax.set_xticks(positions)
            ax.set_xticklabels(xtick_labels, rotation=0, ha="center", fontsize=7.2)
            ax.set_title(f"Panel A1: {task} Jumps" if task == "Can" else f"Panel A2: {task} Jumps",
                         fontsize=9.0, fontweight="bold", pad=6)
            ax.grid(axis="y", linestyle=":", alpha=0.35)

        ax_can.set_ylim(0.0, 0.72)
        ax_can.set_ylabel(r"Commanded Action Jump $\|\Delta \mathbf{a}_t\|_2$", fontsize=8.5)
        ax_sq.tick_params(labelleft=False)

        # --- PANEL B: Expert Switch Rate ---
        b_methods = METHODS_ALL
        y_pos = np.arange(len(b_methods))
        task_offsets = {"Can": -0.14, "Square": 0.14}
        task_colors = {"Can": OKABE_ITO["vermillion"], "Square": OKABE_ITO["blue"]}

        for task in TASKS:
            for i, m in enumerate(b_methods):
                rates = trace_data[task][m]["seed_sw_rates"]
                if not rates:
                    continue
                y = y_pos[i] + task_offsets[task]
                mean_r = float(np.mean(rates))
                min_r = min(rates)
                max_r = max(rates)
                c = task_colors[task]

                # Range bar
                ax_b.plot([min_r, max_r], [y, y], color=c, lw=1.5, zorder=3)
                # Seed mean
                ax_b.scatter([mean_r], [y], color=c, s=32, zorder=4, edgecolor="white", linewidth=0.6)
                # Seed points
                for r in rates:
                    ax_b.scatter([r], [y], facecolors="none", edgecolors=c, s=16, linewidths=0.9, zorder=5)

        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels([registry.display_name(m).replace(" (PF)", "") for m in b_methods], fontsize=8.0)
        ax_b.invert_yaxis()
        ax_b.set_xlabel("Routing Switch Rate", fontsize=8.5)
        ax_b.set_title("Panel B: Switch Rate", fontsize=9.0, fontweight="bold", pad=6)
        ax_b.set_xlim(-0.005, 0.075)
        ax_b.grid(axis="x", linestyle=":", alpha=0.35)

        # Build clean legends
        leg_elements_a = [
            Patch(facecolor=OKABE_ITO["vermillion"], alpha=0.3, edgecolor=OKABE_ITO["vermillion"], label="Switch Step Jump (IQR, 5-95%)"),
            Patch(facecolor=OKABE_ITO["sky"], alpha=0.3, edgecolor=OKABE_ITO["sky"], label="Non-Switch Step Jump (IQR, 5-95%)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="#555555", markersize=5, label="Seed mean jump (n=3)"),
        ]
        leg_elements_b = [
            Line2D([0], [0], marker="o", color=task_colors["Can"], lw=1.5, markersize=5, label="Can (mean + range)"),
            Line2D([0], [0], marker="o", color=task_colors["Square"], lw=1.5, markersize=5, label="Square (mean + range)"),
        ]

        fig.legend(
            handles=leg_elements_a + leg_elements_b,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
            fontsize=7.5,
        )

        fig.subplots_adjust(top=0.86, bottom=0.15, left=0.09, right=0.97, wspace=0.46)
    return save(fig, "figures/main/F4_action_discontinuity")
