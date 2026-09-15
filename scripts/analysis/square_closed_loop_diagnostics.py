"""CPU-only closed-loop diagnostics for the Square regression.

This companion audit uses the fields that are actually present in full
rollout traces: end-effector position summaries, normalized actions, route
selection, router margin, expert disagreement, and nearest-training-state
distance.  It does *not* call the position difference a force measurement.

The central systems question is whether a Square failure is more consistent
with one of four observable failure signatures:

1. representation/data mismatch: large nearest-training distances;
2. routing boundary stress: switches accompanied by low margins and large
   changes in the selected action;
3. action-to-motion mismatch: large commands with little observed EEF motion;
4. expert incompatibility: high disagreement between the two candidate
   expert actions near a routing transition.

The outputs are diagnostic associations, not causal claims.  The archived
traces do not contain force/torque or measured velocity, and the rollout
summary does not identify which individual trace episodes succeeded.  The
script therefore records those limitations explicitly.

Example::

    uv run python scripts/analysis/square_closed_loop_diagnostics.py \
        --outputs final_experiments_results/abalation_final/.../outputs_router_ablation_can_square \
        --tasks Can Square \
        --out outputs_cpu_debug/square_closed_loop_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
_PHYSICAL_MEASUREMENT_KEYS = frozenset(
    {"force", "contact_force", "eef_velocity", "actual_eef_velocity", "contact_state"}
)


def _long_path(path: Path) -> Path:
    resolved = str(path.expanduser().resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?"):
        return Path("\\\\?\\" + resolved)
    return Path(resolved)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))], dtype=np.float64
    )
    if finite.size == 0:
        return {"q25": None, "q50": None, "q75": None, "q95": None}
    q25, q50, q75, q95 = np.quantile(finite, [0.25, 0.50, 0.75, 0.95])
    return {"q25": float(q25), "q50": float(q50), "q75": float(q75), "q95": float(q95)}


def _l2(left: Any, right: Any) -> float | None:
    if not isinstance(left, list) or not isinstance(right, list):
        return None
    try:
        a = np.asarray(left, dtype=np.float64)
        b = np.asarray(right, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        return None
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    return float(np.linalg.norm(a - b))


def _dot_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(child, name))
    return result


def _config_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False}
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {"available": False, "error": str(exc)}
    flat = _flatten(config)
    keys = (
        "models.expert.beta",
        "models.router._target_",
        "models.router.top_k",
        "models.router.noise_std",
        "models.router.normalize_input",
        "models.router.balance_coeff",
        "models.router_init.type",
        "models.router_init.prototype_source",
        "models.router_init.mapping_mode",
        "train.margin.enabled",
        "train.margin.lambda_margin",
        "train.supcon.enabled",
        "train.supcon.zero_ce",
        "train.supcon.label_field",
        "train.phase_label_field",
    )
    return {"available": True, "selected": {key: flat.get(key) for key in keys if key in flat}}


def _discover(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, _subdirectories, filenames in os.walk(str(_long_path(root))):
        if "trace.jsonl" in filenames:
            paths.append(Path(directory) / "trace.jsonl")
    return sorted(paths, key=lambda path: str(path).lower())


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def _episode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row.get("timestep", 0)))
    commands: list[float] = []
    motions: list[float] = []
    response_ratios: list[float] = []
    alignments: list[float] = []
    distances: list[float] = []
    switch_commands: list[float] = []
    switch_motions: list[float] = []
    switch_response: list[float] = []
    switch_alignment: list[float] = []
    switch_distances: list[float] = []
    non_switch_commands: list[float] = []
    non_switch_motions: list[float] = []
    stalled_candidates: list[bool] = []
    physical_keys: set[str] = set()

    for row in ordered:
        raw = row.get("raw_obs_summary")
        if isinstance(raw, dict):
            physical_keys.update(key for key in raw if key in _PHYSICAL_MEASUREMENT_KEYS)
        physical_keys.update(key for key in row if key in _PHYSICAL_MEASUREMENT_KEYS)

    transition_records: list[dict[str, Any]] = []
    for index in range(1, len(ordered)):
        previous = ordered[index - 1]
        current = ordered[index]
        action = current.get("final_action")
        if not isinstance(action, list) or len(action) < 3:
            continue
        try:
            command_vector = np.asarray(action[:3], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if command_vector.shape != (3,) or not np.isfinite(command_vector).all():
            continue
        command_norm = float(np.linalg.norm(command_vector))
        motion = _l2(
            (previous.get("raw_obs_summary") or {}).get("eef_pos"),
            (current.get("raw_obs_summary") or {}).get("eef_pos"),
        )
        if motion is None:
            continue
        ratio = motion / command_norm if command_norm > 1e-12 else None
        alignment = _dot_cosine(
            command_vector,
            np.asarray(
                (current.get("raw_obs_summary") or {}).get("eef_pos"), dtype=np.float64
            )
            - np.asarray(
                (previous.get("raw_obs_summary") or {}).get("eef_pos"), dtype=np.float64
            ),
        )
        distance = _finite(current.get("nearest_train_dist"))
        switched = (
            previous.get("selected_expert") is not None
            and current.get("selected_expert") is not None
            and previous.get("selected_expert") != current.get("selected_expert")
        )
        record = {
            "timestep": int(current.get("timestep", index)),
            "command_norm": command_norm,
            "eef_step_norm": motion,
            "command_response_ratio": ratio,
            "command_motion_alignment_cosine": alignment,
            "nearest_train_dist": distance,
            "switched": switched,
        }
        transition_records.append(record)
        commands.append(command_norm)
        motions.append(motion)
        if ratio is not None:
            response_ratios.append(ratio)
        if alignment is not None:
            alignments.append(alignment)
        if distance is not None:
            distances.append(distance)
        if switched:
            switch_commands.append(command_norm)
            switch_motions.append(motion)
            if ratio is not None:
                switch_response.append(ratio)
            if alignment is not None:
                switch_alignment.append(alignment)
            if distance is not None:
                switch_distances.append(distance)
        else:
            non_switch_commands.append(command_norm)
            non_switch_motions.append(motion)

    if commands and motions:
        command_q75 = float(np.quantile(commands, 0.75))
        motion_q25 = float(np.quantile(motions, 0.25))
        stalled_candidates = [
            record["command_norm"] >= command_q75
            and record["eef_step_norm"] <= motion_q25
            for record in transition_records
        ]
    termination_reasons = {
        str(row.get("termination_reason")) for row in ordered if row.get("termination_reason")
    }
    success = termination_reasons == {"success"}
    return {
        "episode_id": ordered[0].get("episode_id") if ordered else None,
        "steps": len(ordered),
        "success": success,
        "termination_reasons": sorted(termination_reasons),
        "transition_count_with_position_and_action": len(transition_records),
        "switch_transition_count": sum(bool(item["switched"]) for item in transition_records),
        "command_norm": _quantiles(commands),
        "eef_step_norm": _quantiles(motions),
        "command_response_ratio": _quantiles(response_ratios),
        "command_motion_alignment_cosine": _quantiles(alignments),
        "nearest_train_dist": _quantiles(distances),
        "switch_command_norm": _quantiles(switch_commands),
        "switch_eef_step_norm": _quantiles(switch_motions),
        "switch_command_response_ratio": _quantiles(switch_response),
        "switch_alignment_cosine": _quantiles(switch_alignment),
        "switch_nearest_train_dist": _quantiles(switch_distances),
        "non_switch_command_norm": _quantiles(non_switch_commands),
        "non_switch_eef_step_norm": _quantiles(non_switch_motions),
        "candidate_stalled_transition_fraction": (
            float(np.mean(stalled_candidates)) if stalled_candidates else None
        ),
        "candidate_stalled_switch_fraction": (
            float(
                np.mean(
                    [
                        stalled_candidates[index]
                        for index, record in enumerate(transition_records)
                        if record["switched"]
                    ]
                )
            )
            if any(record["switched"] for record in transition_records)
            else None
        ),
        "available_physical_measurement_fields": sorted(physical_keys),
        "position_source": "trace.raw_obs_summary.eef_pos",
        "command_source": "trace.final_action[0:3]",
    }


def _run(path: Path) -> dict[str, Any] | None:
    summary = _json(path.parent / "rollout_summary.json")
    if summary is None:
        return None
    metadata = _json(path.parent / "run_meta.json") or {}
    rows = _trace_rows(path)
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row.get("episode_id", "missing"))].append(row)
    return {
        "run_directory": str(path.parent),
        "task": summary.get("task") or summary.get("tag"),
        "model": summary.get("model"),
        "method": metadata.get("method") or summary.get("method") or summary.get("model"),
        "seed": summary.get("training_seed"),
        "success_rate": (summary.get("metrics") or {}).get("eval/rollout/success_rate"),
        "successes": (summary.get("metrics") or {}).get("eval/rollout/successes"),
        "episodes_reported": summary.get("episodes"),
        "config_contract": _config_contract(path.parent / "resolved_config.yaml"),
        "episodes": [_episode(items) for items in by_episode.values() if items],
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(str(run.get("task")), str(run.get("method")))].append(run)
    result: dict[str, Any] = {}
    for (task, method), items in sorted(grouped.items()):
        episodes = [episode for run in items for episode in run["episodes"]]
        def medians(field: str, nested: str = "q50") -> list[float]:
            values: list[float] = []
            for episode in episodes:
                value = episode.get(field, {}).get(nested)
                number = _finite(value)
                if number is not None:
                    values.append(number)
            return values

        def status_medians(success: bool, field: str, nested: str = "q50") -> list[float]:
            return [
                value
                for episode in episodes
                if episode.get("success") is success
                for value in [
                    _finite((episode.get(field) or {}).get(nested))
                    if isinstance(episode.get(field), dict)
                    else _finite(episode.get(field))
                ]
                if value is not None
            ]

        result[f"{task}::{method}"] = {
            "task": task,
            "method": method,
            "seed_count": len(items),
            "reported_success_rate_by_seed": [run.get("success_rate") for run in items],
            "median_command_norm_by_episode": medians("command_norm"),
            "median_eef_step_norm_by_episode": medians("eef_step_norm"),
            "median_response_ratio_by_episode": medians("command_response_ratio"),
            "median_alignment_by_episode": medians("command_motion_alignment_cosine"),
            "median_nearest_train_dist_by_episode": medians("nearest_train_dist"),
            "median_switch_response_ratio_by_episode": medians(
                "switch_command_response_ratio"
            ),
            "median_switch_alignment_by_episode": medians("switch_alignment_cosine"),
            "stalled_fraction_by_episode": [
                episode["candidate_stalled_transition_fraction"] for episode in episodes
            ],
            "stalled_switch_fraction_by_episode": [
                episode["candidate_stalled_switch_fraction"] for episode in episodes
            ],
            "success_episode_count": sum(bool(episode.get("success")) for episode in episodes),
            "failure_episode_count": sum(not bool(episode.get("success")) for episode in episodes),
            "success_conditioned": {
                "success": {
                    "median_response_ratio": status_medians(True, "command_response_ratio"),
                    "median_alignment": status_medians(True, "command_motion_alignment_cosine"),
                    "median_nearest_train_dist": status_medians(True, "nearest_train_dist"),
                    "stalled_fraction": status_medians(
                        True, "candidate_stalled_transition_fraction", nested="q50"
                    ),
                },
                "failure": {
                    "median_response_ratio": status_medians(False, "command_response_ratio"),
                    "median_alignment": status_medians(False, "command_motion_alignment_cosine"),
                    "median_nearest_train_dist": status_medians(False, "nearest_train_dist"),
                    "stalled_fraction": status_medians(
                        False, "candidate_stalled_transition_fraction", nested="q50"
                    ),
                },
            },
            "physical_measurement_fields": sorted(
                {
                    field
                    for episode in episodes
                    for field in episode["available_physical_measurement_fields"]
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
        "--out",
        type=Path,
        default=Path("outputs_cpu_debug/square_closed_loop_diagnostics.json"),
    )
    args = parser.parse_args(argv)
    tasks = {str(task).lower() for task in args.tasks}
    methods = set(args.methods) if args.methods else None
    runs: list[dict[str, Any]] = []
    for trace in _discover(args.outputs):
        try:
            run = _run(trace)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            continue
        if run is None or str(run.get("task", "")).lower() not in tasks:
            continue
        if methods is not None and str(run.get("method")) not in methods:
            continue
        runs.append(run)
    if not runs:
        print("No valid rollout traces matched the requested tasks/methods.", file=sys.stderr)
        return 2
    report = {
        "analysis": {
            "name": "square_closed_loop_diagnostics",
            "cpu_only": True,
            "repository": str(REPO),
            "outputs_root": str(_long_path(args.outputs)),
            "tasks_requested": list(args.tasks),
            "methods_requested": args.methods,
            "runs_included": len(runs),
            "hypotheses": {
                "representation_or_ood": (
                    "nearest_train_dist is elevated and/or action response differs "
                    "from controls."
                ),
                "boundary_stress": (
                    "switch transitions show larger action discontinuity, lower "
                    "margin, or high expert disagreement."
                ),
                "contact_or_stall_proxy": (
                    "large command coincides with small observed EEF displacement; "
                    "this is not force."
                ),
                "control_direction_mismatch": "command-to-motion alignment is systematically poor.",
            },
        },
        "aggregate": _aggregate(runs),
        "runs": runs,
        "limitations": [
            "EEF position differences are a motion proxy, not measured velocity or contact force.",
            (
                "The archived trace schema does not expose per-episode success labels, "
                "so success-conditioned mechanisms cannot be identified from these "
                "files alone."
            ),
            (
                "A command-to-motion ratio is only interpretable under the "
                "repository's action/EEF coordinate contract; it is not a calibrated "
                "physical compliance metric."
            ),
            (
                "Cross-method comparisons remain descriptive when representation "
                "provider, router package, or initialization differs."
            ),
            (
                "Causal claims require a matched intervention, such as a fixed "
                "router with controlled boundary smoothing or a beta-matched expert "
                "comparison."
            ),
        ],
    }
    destination = args.out.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for key, value in report["aggregate"].items():
        print(
            f"{key}: success={value['reported_success_rate_by_seed']} "
            f"median_response={value['median_response_ratio_by_episode'][:3]} "
            f"median_alignment={value['median_alignment_by_episode'][:3]} "
            f"stalled={value['stalled_fraction_by_episode'][:3]}"
        )
    print(f"[square-closed-loop-diagnostics] report={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
