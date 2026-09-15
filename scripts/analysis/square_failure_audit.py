"""Audit Square rollout regressions without assigning unsupported causality.

This script consumes archived ``trace.jsonl`` and ``rollout_summary.json``
files.  It is deliberately inference-only and CPU-only: it never trains a
model, changes an artifact, or treats a rollout correlation as a causal proof.

The audit separates the following observable factors:

* task/method/seed and rollout success bookkeeping;
* router switching, margin, entropy, and expert disagreement;
* action jumps at switches versus non-switch steps;
* routing balance and proximity of switches to episode termination;
* trace-field availability, especially force and measured velocity fields;
* resolved configuration and provenance fields needed to identify confounds.

It is intended to answer questions such as:

* Is the Square regression specific to topology initialization, or is it
  shared by all hard top-1 prototype routers?
* Are switches associated with larger action jumps or lower margins?
* Are the required physical measurements present to test a jamming claim?
* Are comparisons confounded by beta, margin, router package, or provider?

The output is descriptive.  It must not be cited as evidence that switching
causes failure unless a pre-registered causal design is added.

Example::

    uv run python scripts/analysis/square_failure_audit.py \
        --outputs final_experiments_results/abalation_final/.../outputs_router_ablation_can_square \
        --tasks Can Square \
        --out outputs_cpu_debug/square_failure_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
_SUCCESS_TERMS = frozenset({"success", "task_success", "completed_successfully"})
_PHYSICAL_FIELDS = frozenset(
    {
        "contact_force",
        "force",
        "eef_velocity",
        "actual_eef_velocity",
        "measured_velocity",
        "contact_state",
        "contact",
    }
)


def _long_path(path: Path) -> Path:
    """Use a Windows extended path for deeply nested cloud archives."""
    resolved = str(path.expanduser().resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return Path("\\\\?\\" + resolved)
    return Path(resolved)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[float]) -> float | None:
    items = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(items)) if items else None


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    items = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if items.size == 0:
        return {"q05": None, "q25": None, "q50": None, "q75": None, "q95": None}
    q05, q25, q50, q75, q95 = np.quantile(items, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "q05": float(q05),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "q95": float(q95),
    }


def _l2_delta(previous: Any, current: Any) -> float | None:
    if not isinstance(previous, list) or not isinstance(current, list):
        return None
    try:
        left = np.asarray(previous, dtype=np.float64)
        right = np.asarray(current, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        return None
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        return None
    return float(np.linalg.norm(right - left))


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _flatten_config(config: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(config, dict):
        for key, value in config.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_config(value, child))
    else:
        result[prefix] = config
    return result


def _config_snapshot(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        return {"available": False, "reason": "resolved_config.yaml is missing"}
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {"available": False, "reason": f"could not parse config: {exc}"}
    flat = _flatten_config(config)
    wanted = {
        key: flat[key]
        for key in sorted(flat)
        if any(
            token in key.lower()
            for token in (
                "beta",
                "router",
                "prototype",
                "phase",
                "supcon",
                "margin",
                "top_k",
                "balance",
                "expert_init",
                "stage2_source",
            )
        )
    }
    return {"available": True, "selected": wanted}


def _discover_rollouts(root: Path) -> list[Path]:
    """Discover traces with os.walk so deep Windows paths are not truncated."""
    trace_paths: list[Path] = []
    root_string = str(_long_path(root))
    for directory, _subdirectories, filenames in os.walk(root_string):
        if "trace.jsonl" in filenames:
            trace_paths.append(Path(directory) / "trace.jsonl")
    return sorted(trace_paths, key=lambda path: str(path).lower())


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _trace_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        return [], [f"cannot open trace: {exc}"]
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(item, dict):
                errors.append(f"line {line_number}: row is not an object")
                continue
            rows.append(item)
    return rows, errors


def _row_scalar(row: dict[str, Any], key: str) -> float | None:
    return _finite(row.get(key))


def _terminal_reason(rows: list[dict[str, Any]]) -> str | None:
    reasons = [str(row["termination_reason"]) for row in rows if row.get("termination_reason")]
    return Counter(reasons).most_common(1)[0][0] if reasons else None


def _episode_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: int(row.get("timestep", 0)))
    selected = [row.get("selected_expert") for row in rows]
    switches: list[int] = []
    jumps: list[float] = []
    switch_jumps: list[float] = []
    nonswitch_jumps: list[float] = []
    margins: list[float] = []
    switch_margins: list[float] = []
    nonswitch_margins: list[float] = []
    disagreements: list[float] = []
    switch_disagreements: list[float] = []
    nonswitch_disagreements: list[float] = []
    entropies: list[float] = []
    distances: list[float] = []
    physical_fields: set[str] = set()

    previous_row: dict[str, Any] | None = None
    for row in rows:
        raw_summary = row.get("raw_obs_summary")
        if isinstance(raw_summary, dict):
            physical_fields.update(str(key) for key in raw_summary if str(key) in _PHYSICAL_FIELDS)
        physical_fields.update(str(key) for key in row if str(key) in _PHYSICAL_FIELDS)

        margin = _row_scalar(row, "router_margin")
        if margin is not None:
            margins.append(margin)
        disagreement = _row_scalar(row, "expert_disagreement")
        if disagreement is not None:
            disagreements.append(disagreement)
        entropy = _row_scalar(row, "router_entropy")
        if entropy is not None:
            entropies.append(entropy)
        distance = _row_scalar(row, "nearest_train_dist")
        if distance is not None:
            distances.append(distance)

        if previous_row is not None:
            switched = (
                previous_row.get("selected_expert") is not None
                and row.get("selected_expert") is not None
                and previous_row.get("selected_expert") != row.get("selected_expert")
            )
            if switched:
                switches.append(int(row.get("timestep", 0)))
            jump = _l2_delta(previous_row.get("final_action"), row.get("final_action"))
            if jump is not None:
                jumps.append(jump)
                if switched:
                    switch_jumps.append(jump)
                else:
                    nonswitch_jumps.append(jump)
            if margin is not None:
                (switch_margins if switched else nonswitch_margins).append(margin)
            if disagreement is not None:
                (switch_disagreements if switched else nonswitch_disagreements).append(
                    disagreement
                )
        previous_row = row

    horizon = max((int(row.get("timestep", 0)) for row in rows), default=-1) + 1
    route_counts = Counter(
        int(expert) for expert in selected if isinstance(expert, (int, float))
    )
    route_total = sum(route_counts.values())
    route_probs = np.asarray(
        [count / route_total for count in route_counts.values()], dtype=np.float64
    )
    route_entropy = (
        float(-(route_probs * np.log(route_probs)).sum()) if route_probs.size else None
    )
    terminal = _terminal_reason(rows)
    return {
        "episode_id": rows[0].get("episode_id") if rows else None,
        "steps": len(rows),
        "trace_horizon": horizon,
        "terminal_reason": terminal,
        "terminal_reason_is_explicit_success": terminal in _SUCCESS_TERMS,
        "switch_count": len(switches),
        "switch_rate_per_transition": len(switches) / max(len(rows) - 1, 1),
        "switch_timesteps": switches,
        "steps_from_last_switch_to_trace_end": (
            horizon - switches[-1] if switches else None
        ),
        "action_jump": _quantiles(jumps),
        "switch_action_jump": _quantiles(switch_jumps),
        "non_switch_action_jump": _quantiles(nonswitch_jumps),
        "router_margin": _quantiles(margins),
        "switch_router_margin": _quantiles(switch_margins),
        "non_switch_router_margin": _quantiles(nonswitch_margins),
        "expert_disagreement": _quantiles(disagreements),
        "switch_expert_disagreement": _quantiles(switch_disagreements),
        "non_switch_expert_disagreement": _quantiles(nonswitch_disagreements),
        "router_entropy": _quantiles(entropies),
        "nearest_train_dist": _quantiles(distances),
        "selected_expert_counts": dict(sorted(route_counts.items())),
        "selected_expert_entropy": route_entropy,
        "available_physical_fields": sorted(physical_fields),
    }


def _aggregate_episode_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str, nested: str | None = None) -> list[float]:
        result: list[float] = []
        for episode in episodes:
            value: Any = episode.get(key)
            if nested is not None and isinstance(value, dict):
                value = value.get(nested)
            number = _finite(value)
            if number is not None:
                result.append(number)
        return result

    switch_counts = values("switch_count")
    switch_rates = values("switch_rate_per_transition")
    return {
        "episodes": len(episodes),
        "switch_count_total": int(sum(switch_counts)),
        "switch_rate_mean": _mean(switch_rates),
        "switch_rate_quantiles": _quantiles(switch_rates),
        "action_jump_q50_all": _mean(values("action_jump", "q50")),
        "action_jump_q50_at_switch": _mean(values("switch_action_jump", "q50")),
        "action_jump_q50_non_switch": _mean(values("non_switch_action_jump", "q50")),
        "router_margin_q50_all": _mean(values("router_margin", "q50")),
        "router_margin_q50_at_switch": _mean(values("switch_router_margin", "q50")),
        "router_margin_q50_non_switch": _mean(values("non_switch_router_margin", "q50")),
        "expert_disagreement_q50_at_switch": _mean(
            values("switch_expert_disagreement", "q50")
        ),
        "expert_disagreement_q50_non_switch": _mean(
            values("non_switch_expert_disagreement", "q50")
        ),
        "router_entropy_q50": _mean(values("router_entropy", "q50")),
        "nearest_train_dist_q50": _mean(values("nearest_train_dist", "q50")),
        "episodes_with_explicit_success_reason": sum(
            bool(episode["terminal_reason_is_explicit_success"]) for episode in episodes
        ),
        "physical_fields_observed": sorted(
            {
                field
                for episode in episodes
                for field in episode["available_physical_fields"]
            }
        ),
    }


def _run_report(trace_path: Path) -> dict[str, Any] | None:
    run_dir = trace_path.parent
    summary = _load_json(run_dir / "rollout_summary.json")
    if summary is None:
        return None
    provenance = _load_json(run_dir / "run_meta.json") or {}
    rows, parse_errors = _trace_rows(trace_path)
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row.get("episode_id", "missing"))].append(row)
    episodes = [_episode_metrics(items) for items in by_episode.values() if items]
    metrics = summary.get("metrics") or {}
    return {
        "run_directory": str(run_dir),
        "model": summary.get("model"),
        "method": provenance.get("method") or summary.get("method") or summary.get("model"),
        "task": summary.get("task") or summary.get("tag"),
        "seed": summary.get("training_seed"),
        "episodes_reported": summary.get("episodes"),
        "successes_reported": metrics.get("eval/rollout/successes"),
        "success_rate_reported": metrics.get("eval/rollout/success_rate"),
        "failure_categories_reported": summary.get("failure_categories"),
        "trace_rows": len(rows),
        "trace_parse_errors": parse_errors,
        "config": _config_snapshot(run_dir / "resolved_config.yaml"),
        "provenance": provenance,
        "episodes": episodes,
        "aggregate": _aggregate_episode_metrics(episodes),
    }


def _group_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(str(run.get("task")), str(run.get("method")))].append(run)
    result: dict[str, Any] = {}
    for (task, method), items in sorted(groups.items()):
        success_rates = [
            number
            for number in (_finite(item.get("success_rate_reported")) for item in items)
            if number is not None
        ]
        result[f"{task}::{method}"] = {
            "task": task,
            "method": method,
            "seed_count": len(items),
            "reported_success_rate_mean": _mean(success_rates),
            "reported_success_rate_by_seed": success_rates,
            "switch_rate_mean_by_seed": [
                item["aggregate"].get("switch_rate_mean") for item in items
            ],
            "action_jump_q50_at_switch_by_seed": [
                item["aggregate"].get("action_jump_q50_at_switch") for item in items
            ],
            "action_jump_q50_non_switch_by_seed": [
                item["aggregate"].get("action_jump_q50_non_switch") for item in items
            ],
            "router_margin_q50_at_switch_by_seed": [
                item["aggregate"].get("router_margin_q50_at_switch") for item in items
            ],
            "expert_disagreement_q50_at_switch_by_seed": [
                item["aggregate"].get("expert_disagreement_q50_at_switch") for item in items
            ],
            "physical_fields_observed": sorted(
                {
                    field
                    for item in items
                    for field in item["aggregate"].get("physical_fields_observed", [])
                }
            ),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["Can", "Square"])
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs_cpu_debug/square_failure_audit.json")
    )
    args = parser.parse_args(argv)

    output_root = _long_path(args.outputs)
    traces = _discover_rollouts(output_root)
    if not traces:
        print(f"No trace.jsonl files found below {output_root}", file=sys.stderr)
        return 2

    allowed_tasks = {str(task).lower() for task in args.tasks}
    allowed_methods = {str(method) for method in args.methods} if args.methods else None
    runs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for trace in traces:
        run = _run_report(trace)
        if run is None:
            skipped.append(
                {"trace": str(trace), "reason": "missing or invalid rollout_summary.json"}
            )
            continue
        task = str(run.get("task", ""))
        method = str(run.get("method", ""))
        if task.lower() not in allowed_tasks:
            continue
        if allowed_methods is not None and method not in allowed_methods:
            continue
        runs.append(run)

    if not runs:
        print("No valid runs matched --tasks/--methods.", file=sys.stderr)
        return 2

    report = {
        "analysis": {
            "name": "square_failure_audit",
            "cpu_only": True,
            "repository": str(REPO),
            "outputs_root": str(output_root),
            "tasks_requested": list(args.tasks),
            "methods_requested": args.methods,
            "runs_included": len(runs),
            "trace_files_discovered": len(traces),
            "physical_fields_required_for_force_claim": sorted(_PHYSICAL_FIELDS),
        },
        "group_summary": _group_summary(runs),
        "runs": runs,
        "skipped": skipped,
        "interpretation_guardrails": [
            (
                "Success rates are copied from rollout_summary.json and are not "
                "recomputed from incomplete traces."
            ),
            (
                "Switch/action associations are descriptive and do not establish "
                "that switching causes failure."
            ),
            (
                "A force or jamming claim requires measured force/contact or actual "
                "velocity fields; their absence is reported explicitly."
            ),
            (
                "Comparisons across router packages or representation providers are "
                "not router-only causal comparisons."
            ),
            (
                "Three seeds support descriptive uncertainty reporting, not a "
                "universal significance claim."
            ),
        ],
    }
    destination = args.out.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for key, summary in report["group_summary"].items():
        print(
            f"{key}: seeds={summary['seed_count']} "
            f"success_mean={summary['reported_success_rate_mean']!r} "
            f"switch_rate={summary['switch_rate_mean_by_seed']} "
            f"switch_jump={summary['action_jump_q50_at_switch_by_seed']} "
            f"non_switch_jump={summary['action_jump_q50_non_switch_by_seed']}"
        )
    print(f"[square-failure-audit] report={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
