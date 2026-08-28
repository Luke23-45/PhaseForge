"""Publication style contract (figures_tables_plan.md section 5).

Okabe-Ito colorblind-safe palette, one fixed method->color mapping across
every figure, PhaseForge accent-first ordering, NeurIPS column widths,
vector PDF + 300 DPI PNG exports with embedded fonts.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # analysis runs headless; set before pyplot import
import matplotlib as mpl
import matplotlib.pyplot as plt

from studies.analysis.common.config import get_config

# Okabe-Ito colorblind-safe palette.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
    "dark_yellow": "#a08600",  # legible yellow substitute for lines on white
}

# Fixed method -> color mapping: PhaseForge carries the accent; controls are
# muted; the BC family shares cool neutrals; ablation groups have distinct palettes.
METHOD_COLORS: dict[str, str] = {
    "phaseforge": OKABE_ITO["vermillion"],
    "bc": OKABE_ITO["blue"],
    "bc_large": OKABE_ITO["sky"],
    "bc_robot_only": OKABE_ITO["grey"],
    "scratch_moe": OKABE_ITO["green"],
    "warmstart_moe": OKABE_ITO["orange"],
    "phase_pretrain_random_router": OKABE_ITO["dark_yellow"],
    "plain_encoder_phase_bootstrap": OKABE_ITO["black"],
    "teacher_forced": OKABE_ITO["grey"],
    # Router initialization family (Group A)
    "pf_spherical_kmeans": OKABE_ITO["sky"],
    "pf_kmeans": OKABE_ITO["blue"],
    "pf_phase_head": OKABE_ITO["green"],
    "pf_random_random": OKABE_ITO["orange"],
    "pf_centroid_random": OKABE_ITO["purple"],
    "pf_spherical": OKABE_ITO["vermillion"],
    "pf_ft": OKABE_ITO["green"],
    # Capacity scaling (Group D)
    "pf_k3": OKABE_ITO["sky"],
    "pf_k12": OKABE_ITO["purple"],
    # Phase noise / corruption (Group C)
    "pf_corrupt_25": OKABE_ITO["orange"],
    "pf_corrupt_50": OKABE_ITO["dark_yellow"],
    "pf_shuffle_control": OKABE_ITO["grey"],
    # Expert init suite (Group B)
    "pf_one_warm_plus_random": OKABE_ITO["purple"],
    "pf_full_warm": OKABE_ITO["orange"],
    "pf_drop00": OKABE_ITO["sky"],
    "pf_drop25": OKABE_ITO["blue"],
    "pf_drop50": OKABE_ITO["vermillion"],
    "pf_drop75": OKABE_ITO["orange"],
    "pf_drop100": OKABE_ITO["grey"],
}
ABLATION_COLOR = OKABE_ITO["grey"]
SEED_POINT_COLOR = OKABE_ITO["black"]

FONT_SIZES = {"title": 10, "label": 9, "tick": 8, "legend": 8, "annot": 7.5}


def method_color(name: str) -> str:
    return METHOD_COLORS.get(name, ABLATION_COLOR)


METHOD_MARKERS: dict[str, str] = {
    "phaseforge": "o",
    "bc": "s",
    "bc_large": "D",
    "scratch_moe": "v",
    "warmstart_moe": "P",
    "phase_pretrain_random_router": "X",
    "plain_encoder_phase_bootstrap": "p",
    "pf_spherical_kmeans": "s",
    "pf_kmeans": "^",
    "pf_phase_head": "D",
    "pf_random_random": "x",
}


def method_marker(name: str) -> str:
    return METHOD_MARKERS.get(name, "o")


def column_width(kind: str = "text") -> float:
    cfg = get_config().style
    if kind == "text":
        return float(cfg.text_width_in)
    if kind == "margin":
        return float(cfg.margin_width_in)
    raise ValueError(f"Unknown column kind {kind!r}")


@contextmanager
def paper_style() -> Iterator[None]:
    """Apply the publication rcParams for the enclosed figure block."""
    with mpl.rc_context(
        {
            "figure.dpi": 100,
            "savefig.dpi": int(get_config().style.dpi),
            "font.size": FONT_SIZES["tick"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["label"],
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
            "legend.fontsize": FONT_SIZES["legend"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        yield


def save_figure(fig: plt.Figure, out_base: Path) -> list[Path]:
    """Save one figure as vector PDF + raster PNG next to out_base (no suffix)."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    pdf = out_base.with_suffix(".pdf")
    png = out_base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]
