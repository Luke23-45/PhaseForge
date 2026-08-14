"""Canonical row schema for PhaseForge evaluation results.

Every entry in :file:`outputs/_results/results.jsonl` is validated by
:func:`validate_row` before it is appended. The validator rejects missing
required keys, wrong-typed values, and unknown top-level keys; optional
metric columns are type-checked and inf-rejected (NaN permitted) only when
present so that ``bc`` runs (no router metrics) stay schema-valid.

The forward-compatibility surface is the ``extra`` dict — any keys that
do not fit the canonical schema go there rather than being silently
dropped or rejected.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

_CORE_REQUIRED = (
    "run_id",
    "timestamp",
    "model",
    "stage",
    "seed",
    "git_sha",
    "config_hash",
    "device",
    "ckpt_path",
    "action_mse",
)

_INT_FIELDS = ("stage", "seed")
_STR_FIELDS = (
    "run_id",
    "timestamp",
    "model",
    "git_sha",
    "config_hash",
    "device",
    "ckpt_path",
)

# Optional per-metric columns. Validator type-checks these only when they
# are present so that methods without a router (e.g. ``bc``) can produce
# schema-valid rows.
OPTIONAL_METRIC_FIELDS: tuple[str, ...] = (
    "action_l2_threshold_rate",
    "boundary_action_smoothness",
    "routing_entropy",
    "routing_entropy_variance",
    "time_to_stable_routing",
    "routing_stability_fraction",
    "topk_balance_score",
    "top1_balance_score",
    "topk_collapse_rate",
    "top1_collapse_rate",
    "phase_expert_nmi",
)


@dataclass
class ResultRow:
    """One evaluation row = one (model, stage, seed) offline evaluation."""

    run_id: str
    timestamp: str
    model: str
    stage: int
    seed: int
    git_sha: str
    config_hash: str
    device: str
    ckpt_path: str
    action_mse: float
    action_l2_threshold_rate: float = float("nan")
    boundary_action_smoothness: float = float("nan")
    routing_entropy: float = float("nan")
    routing_entropy_variance: float = float("nan")
    time_to_stable_routing: float = float("nan")
    routing_stability_fraction: float = float("nan")
    topk_balance_score: float = float("nan")
    top1_balance_score: float = float("nan")
    topk_collapse_rate: float = float("nan")
    top1_collapse_rate: float = float("nan")
    phase_expert_nmi: float = float("nan")
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SchemaError(ValueError):
    """Raised when a row fails schema validation."""


def _is_real_int(value: Any) -> bool:
    """``bool`` is technically ``int`` in Python — exclude it."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real_numeric(value: Any) -> bool:
    """Numeric (int or float, excluding ``bool``). NaN allowed, inf rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and math.isinf(value):
        return False
    return True


def validate_row(row: dict[str, Any]) -> None:
    """Strict validator. Raises :class:`SchemaError` on any deviation.

    Rules:

    * all required keys present
    * ``stage`` and ``seed`` are ints (``bool`` rejected)
    * string fields are ``str``
    * ``action_mse`` is finite-or-NaN numeric
    * optional metric fields: when present, must be finite-or-NaN numeric
    * ``extra`` (when present) is a dict
    * unknown top-level keys are rejected; put forward-compat fields in ``extra``
    """
    if not isinstance(row, dict):
        raise SchemaError(f"Row must be a dict, got {type(row).__name__}")
    missing = [k for k in _CORE_REQUIRED if k not in row]
    if missing:
        raise SchemaError(f"Row missing required keys: {missing}")
    known = set(_CORE_REQUIRED) | set(OPTIONAL_METRIC_FIELDS) | {"extra"}
    unknown = sorted(set(row) - known)
    if unknown:
        raise SchemaError(f"Row has unknown top-level keys: {unknown}")
    for key in _INT_FIELDS:
        if not _is_real_int(row[key]):
            raise SchemaError(
                f"Row[{key!r}] must be int, got {type(row[key]).__name__}"
            )
    for key in _STR_FIELDS:
        if not isinstance(row[key], str):
            raise SchemaError(
                f"Row[{key!r}] must be str, got {type(row[key]).__name__}"
            )
    if "action_mse" in row and not _is_real_numeric(row["action_mse"]):
        raise SchemaError(
            f"Row['action_mse'] must be finite or NaN, got {type(row['action_mse']).__name__}"
        )
    for key in OPTIONAL_METRIC_FIELDS:
        if key not in row:
            continue
        if not _is_real_numeric(row[key]):
            raise SchemaError(
                f"Row[{key!r}] must be finite or NaN, got {type(row[key]).__name__}"
            )
    if "extra" in row and not isinstance(row["extra"], dict):
        raise SchemaError(
            f"Row['extra'] must be dict, got {type(row['extra']).__name__}"
        )


__all__ = [
    "ResultRow",
    "SchemaError",
    "validate_row",
    "OPTIONAL_METRIC_FIELDS",
]
