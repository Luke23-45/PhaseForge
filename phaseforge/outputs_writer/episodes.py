"""Rollout episode records and their summary statistics.

``eval/{model}/{ts}_{runid}/episodes.jsonl`` holds one append-only,
schema-validated record per **attempted** rollout episode (final
specification §5.4). Infrastructure failures, policy exceptions, invalid
actions and simulator errors are never converted into failed task episodes:
they are counted separately and invalidate the run until rerun.

This module implements the record schema, the append-only writer, the
reader, and the pure statistics that feed ``rollout_success.csv`` and
``rollout_comparisons.csv``. It is deliberately independent of any
simulator: the rollout adapter itself is wired in only after the
simulator/version-parity gate (spec §9.5).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from filelock import FileLock

from phaseforge.outputs_writer.schema import SchemaError

#: Required fields on every episode record.
EPISODE_REQUIRED: tuple[str, ...] = (
    "run_id",
    "model",
    "checkpoint_sha256",
    "task",
    "training_seed",
    "reset_seed",
    "episode_index",
    "valid_episode",
)

_EPISODE_BOOL = ("valid_episode", "success", "timed_out")
_EPISODE_INT = ("training_seed", "reset_seed", "episode_index", "steps")
_EPISODE_STR = ("run_id", "model", "checkpoint_sha256", "task")
_EPISODE_STR_NULLABLE = (
    "termination_reason",
    "failure_category",
    "exception",
    "tag",
)


def _is_real_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_episode_record(row: dict[str, Any]) -> None:
    """Strict validator for one ``episodes.jsonl`` row.

    Rules (spec §5.4):

    * ``success`` is present only for a valid completed episode (a valid
      record must carry it; an invalid one must not).
    * ``failure_category`` is required only for a valid failure and must
      come from a frozen taxonomy (any string here; the taxonomy itself is
      enforced by the rollout adapter).
    * Unknown top-level keys are rejected.
    """
    if not isinstance(row, dict):
        raise SchemaError(f"Episode record must be a dict, got {type(row).__name__}")
    missing = [k for k in EPISODE_REQUIRED if k not in row]
    if missing:
        raise SchemaError(f"Episode record missing required keys: {missing}")
    known = set(EPISODE_REQUIRED) | set(_EPISODE_BOOL) | set(_EPISODE_INT) | set(
        _EPISODE_STR_NULLABLE
    ) | {"extra"}
    unknown = sorted(set(row) - known)
    if unknown:
        raise SchemaError(f"Episode record has unknown top-level keys: {unknown}")

    for key in _EPISODE_STR:
        if not isinstance(row[key], str):
            raise SchemaError(
                f"Episode record[{key!r}] must be str, got {type(row[key]).__name__}"
            )
    for key in _EPISODE_INT:
        if key not in row:
            continue
        if not _is_real_int(row[key]):
            raise SchemaError(
                f"Episode record[{key!r}] must be int, got {type(row[key]).__name__}"
            )
    for key in _EPISODE_BOOL:
        if key not in row:
            continue
        if not isinstance(row[key], bool):
            raise SchemaError(
                f"Episode record[{key!r}] must be bool, got {type(row[key]).__name__}"
            )
    for key in _EPISODE_STR_NULLABLE:
        if key not in row or row[key] is None:
            continue
        if not isinstance(row[key], str):
            raise SchemaError(
                f"Episode record[{key!r}] must be str or null, "
                f"got {type(row[key]).__name__}"
            )

    valid = row["valid_episode"]
    has_success = "success" in row
    if valid and not has_success:
        raise SchemaError(
            "Episode record is valid but missing the required 'success' field"
        )
    if not valid and has_success:
        raise SchemaError(
            "Episode record is invalid (infrastructure failure) but carries "
            "'success' — infra failures must not be converted into task outcomes"
        )
    if valid and not row.get("success") and "failure_category" not in row:
        raise SchemaError(
            "Valid failed episode is missing the required 'failure_category'"
        )
    if not valid and "failure_category" in row:
        raise SchemaError(
            "Invalid episode records (infra failures) must not carry a "
            "'failure_category'"
        )
    if "extra" in row and not isinstance(row["extra"], dict):
        raise SchemaError(
            f"Episode record['extra'] must be dict, got {type(row['extra']).__name__}"
        )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def append_episode_record(output_dir: str | Path, row: dict[str, Any]) -> Path:
    """Validate and append one episode record to ``episodes.jsonl``.

    Raises:
        SchemaError: ``row`` is malformed; no file write occurs.
        OSError: filesystem error.
    """
    try:
        validate_episode_record(row)
    except Exception:
        raise

    output_dir = Path(output_dir)
    target = output_dir / "episodes.jsonl"
    lock = FileLock(str(output_dir / ".episodes.lock"))
    payload = json.dumps(row, default=_json_default) + "\n"
    with lock:
        with open(target, "a", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    return target


def read_episode_records(output_dir: str | Path) -> list[dict[str, Any]]:
    """Read every episode record from an eval run directory (tolerant)."""
    output_dir = Path(output_dir)
    target = output_dir / "episodes.jsonl"
    if not target.exists():
        return []
    text = target.read_text(encoding="utf-8")
    ends_with_newline = text.endswith(("\n", "\r"))
    lines = text.splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        is_last = index == len(lines) - 1
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError:
            if is_last and not ends_with_newline:
                continue
            raise
    return rows


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a success rate ``k/n``.

    Returns ``(lower, upper)``. With ``n == 0`` the interval is ``NaN`` —
    there is no rate to bound.
    """
    if n <= 0:
        return float("nan"), float("nan")
    rate = k / n
    if rate in (0.0, 1.0):
        return rate, rate
    z_sq = z * z
    denom = 1.0 + z_sq / n
    centre = (rate + z_sq / (2.0 * n)) / denom
    margin = z * math.sqrt(rate * (1.0 - rate) / n + z_sq / (4.0 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def summarize_episodes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per ``(task, model, tag, training_seed)`` success rows for the paper.

    ``tag`` keeps data-variant runs that share a ``model`` name (e.g. the
    ``bc`` floor and the ``bc``/``robot_only`` negative control) in separate
    rows, mirroring the offline aggregate tables.

    Each output row carries valid/success counts, the success rate over
    valid episodes, the Wilson interval, and the number of invalid
    (infrastructure-failure) attempts — which invalidate the run until
    rerun and are reported, never folded into the rate.
    """
    groups: dict[tuple[str, str, str | None, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["task"], row["model"], row.get("tag"), int(row["training_seed"]))
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2] or "", k[3])):
        group = groups[key]
        task, model, tag, seed = key
        valid = [r for r in group if r["valid_episode"]]
        successes = sum(1 for r in valid if r.get("success"))
        invalid = len(group) - len(valid)
        low, high = wilson_interval(successes, len(valid))
        out.append(
            {
                "task": task,
                "model": model,
                "tag": tag,
                "training_seed": seed,
                "valid_episodes": len(valid),
                "successes": successes,
                "success_rate": (successes / len(valid)) if valid else float("nan"),
                "wilson_ci95_low": low,
                "wilson_ci95_high": high,
                "invalid_attempts": invalid,
            }
        )
    return out


def paired_rollout_comparisons(
    rows: list[dict[str, Any]],
    *,
    baseline: str = "phaseforge",
) -> list[dict[str, Any]]:
    """Paired PhaseForge-minus-baseline differences per ``(task, tag, seed)``.

    Pairs the success rate of every non-baseline ``(model, tag)`` identity
    against the baseline model on the same task and training seed (the
    protocol shares seeds across variants, so the pairing is exact). Only
    task/seed cells where both identities have valid episodes are emitted.
    Tagged variants of the baseline itself are never paired against it.
    """
    summaries = {
        (s["task"], s["model"], s["tag"], int(s["training_seed"])): s
        for s in summarize_episodes(rows)
    }
    tasks = {key[0] for key in summaries}
    seeds = {int(key[3]) for key in summaries}
    identities = sorted(
        {(key[1], key[2]) for key in summaries},
        key=lambda i: (i[0], i[1] or ""),
    )
    baseline_identity = (baseline, None)

    out: list[dict[str, Any]] = []
    for task in sorted(tasks):
        for seed in sorted(seeds):
            base = summaries.get((task, baseline, None, seed))
            if base is None or not base["valid_episodes"]:
                continue
            for model, tag in identities:
                if (model, tag) == baseline_identity:
                    continue
                other = summaries.get((task, model, tag, seed))
                if other is None or not other["valid_episodes"]:
                    continue
                out.append(
                    {
                        "task": task,
                        "training_seed": seed,
                        "baseline": baseline,
                        "model": model,
                        "tag": tag,
                        "baseline_success_rate": base["success_rate"],
                        "model_success_rate": other["success_rate"],
                        "diff": other["success_rate"] - base["success_rate"],
                    }
                )
    return out


__all__ = [
    "EPISODE_REQUIRED",
    "validate_episode_record",
    "append_episode_record",
    "read_episode_records",
    "wilson_interval",
    "summarize_episodes",
    "paired_rollout_comparisons",
]
