"""F1 — PhaseForge architectural overview and methodology schematic.

Renders a publication-grade vector overview showing:
1. Stage 1: Phase-Supervised Representation Pretraining (Encoder + Phase Head + Action Head)
2. Bootstrap Instant (t=0): Centroid Extraction & Router Init + 50% Parameter Drop (R50)
3. Stage 2: Specialized MoE Policy Deployment with Top-2 Routing
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from studies.analysis.common.style import OKABE_ITO, paper_style
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.figures import save


def generate(dataset: AnalysisDataset | None = None) -> list[Path]:
    with paper_style():
        fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=300)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 50)
        ax.axis("off")

        # Color definitions
        c_stage1 = "#EBF4FA"
        c_stage1_border = OKABE_ITO["blue"]
        c_boot = "#FEF6E9"
        c_boot_border = OKABE_ITO["orange"]
        c_stage2 = "#F2F9F4"
        c_stage2_border = OKABE_ITO["green"]

        # Background Containers
        # Stage 1 Box
        box_s1 = patches.FancyBboxPatch(
            (1, 2), 30, 45, boxstyle="round,pad=0.5,rounding_size=1.5",
            facecolor=c_stage1, edgecolor=c_stage1_border, linewidth=1.2, linestyle="--"
        )
        ax.add_patch(box_s1)
        ax.text(16, 44.5, "Stage 1: Phase Pretraining", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=c_stage1_border)

        # Bootstrap Box
        box_boot = patches.FancyBboxPatch(
            (34, 2), 29, 45, boxstyle="round,pad=0.5,rounding_size=1.5",
            facecolor=c_boot, edgecolor=c_boot_border, linewidth=1.2, linestyle="--"
        )
        ax.add_patch(box_boot)
        ax.text(48.5, 44.5, "Bootstrap Instant (t=0)", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=c_boot_border)

        # Stage 2 Box
        box_s2 = patches.FancyBboxPatch(
            (66, 2), 33, 45, boxstyle="round,pad=0.5,rounding_size=1.5",
            facecolor=c_stage2, edgecolor=c_stage2_border, linewidth=1.2, linestyle="--"
        )
        ax.add_patch(box_s2)
        ax.text(82.5, 44.5, "Stage 2: MoE Policy", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=c_stage2_border)

        # --- Stage 1 Components ---
        # State input
        _draw_node(ax, 16, 38, 22, 4.5, r"State Input $\mathbf{s}_t \in \mathbb{R}^d$" + "\n(robot + object state)", "#FFFFFF", "#555555")
        # Encoder
        _draw_node(ax, 16, 28, 22, 6.0, r"State Encoder $f_\theta$" + "\n(3-layer MLP + Residual)", "#D9EAF7", c_stage1_border)
        _draw_arrow(ax, (16, 35.5), (16, 31.2))

        # Latent z
        ax.text(16, 23.5, r"Latent $\mathbf{z}_t \in \mathbb{R}^{128}$", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#333333")
        _draw_arrow(ax, (16, 25.0), (16, 22.2))

        # Dual Heads
        _draw_node(ax, 8.5, 13.5, 12, 6.5, r"Phase Head $g_\psi$" + "\n" + r"$\mathcal{L}_\mathrm{phase}$ (Cross-Ent)", "#FFFFFF", OKABE_ITO["vermillion"])
        _draw_node(ax, 23.5, 13.5, 12, 6.5, r"Action Head $h_\phi$" + "\n" + r"$\mathcal{L}_\mathrm{action}$ (MSE)", "#FFFFFF", OKABE_ITO["purple"])
        _draw_arrow(ax, (13, 22.0), (8.5, 17.0))
        _draw_arrow(ax, (19, 22.0), (23.5, 17.0))

        # --- Transition Arrow 1 -> Boot ---
        _draw_arrow(ax, (31.5, 25), (33.5, 25), lw=1.8, color="#555555")

        # --- Bootstrap Components ---
        # Centroid Extraction
        _draw_node(ax, 48.5, 36, 24, 7.0, r"1. Latent Clustering" + "\n" + r"$\mathbf{c}_k = \frac{1}{|D_k|} \sum_{i \in D_k} f_\theta(\mathbf{s}_i)$", "#FFFFFF", c_boot_border)

        # Router Init
        _draw_node(ax, 48.5, 24, 24, 6.0, r"2. Prototype Router Init" + "\n" + r"$\mathbf{W}_R = [\mathbf{c}_1, \dots, \mathbf{c}_K]^\top$", "#FFF2DE", OKABE_ITO["vermillion"])
        _draw_arrow(ax, (48.5, 32.2), (48.5, 27.2))

        # Expert Warm-Start
        _draw_node(ax, 48.5, 11.5, 24, 7.5, r"3. Partial Warm-Start (R50)" + "\n" + r"$\mathbf{W}_{E_k} = \mathbf{W}_\mathrm{action} \odot \mathbf{m}_k$" + "\n(50% parameter drop rate)", "#FFF2DE", OKABE_ITO["purple"])
        _draw_arrow(ax, (48.5, 20.8), (48.5, 15.5))

        # --- Transition Arrow Boot -> 2 ---
        _draw_arrow(ax, (63.5, 25), (65.5, 25), lw=1.8, color="#555555")

        # --- Stage 2 Components ---
        # State & Encoder
        _draw_node(ax, 82.5, 37.5, 24, 5.5, r"State $\mathbf{s}_t \rightarrow$ Encoder $f_\theta(\mathbf{s}_t)$" + "\n(Frozen / Fine-tuned)", "#E2F0D9", c_stage2_border)

        # Router
        _draw_node(ax, 74.0, 26.0, 13, 6.5, "Centroid Router\nTop-2 Gating $(w_k)$", "#FFFFFF", OKABE_ITO["vermillion"])
        _draw_arrow(ax, (78.0, 34.5), (74.0, 29.5))

        # Experts
        _draw_node(ax, 90.5, 26.0, 14, 6.5, r"Specialized Experts" + "\n" + r"$E_1, \dots, E_K$ (R50)", "#FFFFFF", OKABE_ITO["purple"])
        _draw_arrow(ax, (87.0, 34.5), (90.5, 29.5))

        # Gating flow
        _draw_arrow(ax, (74.0, 22.5), (82.5, 17.5))
        _draw_arrow(ax, (90.5, 22.5), (82.5, 17.5))

        # Mixture Action Output
        _draw_node(ax, 82.5, 11.5, 26, 8.0, r"Mixture Policy Action Output" + "\n" + r"$\mathbf{a}_t = \sum_{k \in \mathrm{Top\text{-}2}} w_k(\mathbf{z}_t) E_k(\mathbf{z}_t)$" + "\n" + r"$\mathcal{L}_\mathrm{total} = \mathcal{L}_\mathrm{action} + \lambda_\mathrm{bal} \mathcal{L}_\mathrm{balance}$", "#E2F0D9", c_stage2_border)

        fig.tight_layout(pad=0.2)
    return save(fig, "figures/main/F1_overview")


def _draw_node(ax, cx, cy, w, h, text, facecolor, edgecolor):
    box = patches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.0, zorder=3
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=7.2, zorder=4, color="#111111")


def _draw_arrow(ax, start, end, lw=1.2, color="#333333"):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=1, shrinkB=1),
        zorder=5
    )

