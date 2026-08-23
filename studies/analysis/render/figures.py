"""Matplotlib render engine: figure factory + shared publication primitives.

Every figure asset composes these primitives inside ``paper_style()`` and
saves through ``save()`` so styling stays in one place (plan section 5).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from studies.analysis.common.config import paper_root
from studies.analysis.common.style import (
    OKABE_ITO,
    save_figure,
)


def new_figure(width_kind: str = "text", height_in: float = 2.8, ncols: int = 1):
    """Create a styled figure; call within the generator (style applied on save)."""
    from studies.analysis.common.style import column_width

    width = column_width(width_kind) * ncols
    fig, ax = plt.subplots(figsize=(width, height_in))
    return fig, ax


def new_figure_grid(width_kind: str, height_in: float, nrows: int, ncols: int):
    from studies.analysis.common.style import column_width

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(column_width(width_kind), height_in),
        squeeze=False,
    )
    return fig, axes


def save(fig: plt.Figure, relative: str) -> list[Path]:
    """Save into paper_root/<relative-without-suffix> as PDF + PNG.

    Generators draw inside ``paper_style()``; this only persists the figure.
    """
    base = paper_root() / relative
    return save_figure(fig, base)


def forest(
    ax,
    labels: Sequence[str],
    points: Sequence[float],
    lows: Sequence[float],
    highs: Sequence[float],
    colors: Sequence[str] | None = None,
    xlabel: str = "",
) -> None:
    """Dot-and-interval rows; the plan's encoding for paired deltas and CIs."""
    y = range(len(labels))
    ax.axvline(0.0, color=OKABE_ITO["grey"], linewidth=0.8, linestyle="--", zorder=1)
    for i, (p, lo, hi) in enumerate(zip(points, lows, highs)):
        color = colors[i] if colors else OKABE_ITO["blue"]
        ax.plot([lo, hi], [i, i], color=color, linewidth=1.8, zorder=2)
        ax.scatter([p], [i], color=color, s=28, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(list(labels))
    ax.invert_yaxis()
    if xlabel:
        ax.set_xlabel(xlabel)


def plot_seed_trajectories(
    ax,
    per_seed_series: list[list[tuple[float, float]]],
    color: str,
    label: str | None = None,
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """Thin per-seed lines (plan rule: seed points are always drawn)."""
    for series in per_seed_series:
        xs = [p[0] for p in series]
        ys = [p[1] for p in series]
        ax.plot(xs, ys, color=color, alpha=0.35, linewidth=0.9)
    # mean trajectory on the shared union grid
    from studies.analysis.stats.trajectories import common_grid, resample

    grid = common_grid(per_seed_series)
    if grid and per_seed_series:
        means = [
            sum(vals) / len(vals) for vals in zip(*(resample(s, grid) for s in per_seed_series))
        ]
        ax.plot(grid, means, color=color, linewidth=1.8, label=label)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def heatmap(
    ax,
    matrix,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    annotate: bool = True,
    fmt: str = "{:.2f}",
) -> None:
    import numpy as np

    data = np.asarray(matrix, dtype=float)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                if np.isnan(value):
                    continue
                dark = (value - (vmin if vmin is not None else np.nanmin(data))) / max(
                    1e-9,
                    (vmax if vmax is not None else np.nanmax(data))
                    - (vmin if vmin is not None else np.nanmin(data)),
                )
                ax.text(
                    j,
                    i,
                    fmt.format(value),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if dark > 0.55 else "black",
                )
    return im


def ecdf(ax, values: Sequence[float], color: str, label: str | None = None) -> None:
    import numpy as np

    data = np.sort(np.asarray(values, dtype=float))
    if data.size == 0:
        return
    ys = np.arange(1, data.size + 1) / data.size
    ax.step(
        np.concatenate([data, data[-1:]]),
        np.concatenate([ys, [1.0]]),
        where="post",
        color=color,
        linewidth=1.6,
        label=label,
    )


def stacked_bars(
    ax,
    row_labels: Sequence[str],
    shares: dict[str, list[float]],
    colors: dict[str, str] | None = None,
) -> None:
    """100% stacked horizontal bars (outcome/failure categories)."""
    import numpy as np

    categories = list(shares)
    colors = colors or {}
    palette = [colors.get(c, OKABE_ITO["grey"]) for c in categories]
    data = np.asarray([shares[c] for c in categories], dtype=float)
    left = np.zeros(len(row_labels))
    for row, cat in enumerate(categories):
        ax.barh(
            list(range(len(row_labels))),
            data[row],
            left=left,
            color=palette[row],
            label=cat,
            height=0.6,
        )
        left += data[row]
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(list(row_labels))
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _pos: f"{x:.0%}")
