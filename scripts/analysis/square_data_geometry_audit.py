"""Audit dataset geometry that may make Square routing brittle.

This CPU-only analysis operates on the sequence-length-one processed cache,
not on a trained checkpoint.  It tests whether phase or topology partitions
coincide with large action changes or erase substantial within-regime action
variation.  Those are necessary diagnostics for a routing hypothesis because
the router only sees the instantaneous state while the expert must still
produce fine-grained actions.

The script reports associations only.  It does not claim that a labeler,
representation, or router caused a rollout failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analysis.phase_router_diagnosis import (
    _action_means_diagnostic,
    _cache_alignment,
    _candidate_caches,
    _choose_cache,
    _load_cache,
)

REPO = Path(__file__).resolve().parents[2]


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"q25": None, "q50": None, "q75": None, "q95": None}
    q25, q50, q75, q95 = np.quantile(np.asarray(values), [0.25, 0.50, 0.75, 0.95])
    return {"q25": float(q25), "q50": float(q50), "q75": float(q75), "q95": float(q95)}


def _finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if np.isfinite(float(value))]


def _transition_data(
    states: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
    trajectory_ids: np.ndarray,
) -> dict[str, Any]:
    state_deltas: list[float] = []
    action_deltas: list[float] = []
    same_label_actions: list[float] = []
    changed_label_actions: list[float] = []
    local_slopes: list[float] = []
    same_label_slopes: list[float] = []
    changed_label_slopes: list[float] = []
    for trajectory in np.unique(trajectory_ids):
        indices = np.flatnonzero(trajectory_ids == trajectory)
        if indices.size < 2:
            continue
        # Cache files are ordered trajectories; sorting by source index keeps
        # the audit explicit if a future loader changes concatenation order.
        indices = np.sort(indices)
        for left, right in zip(indices[:-1], indices[1:]):
            state_delta = float(np.linalg.norm(states[right] - states[left]))
            action_delta = float(np.linalg.norm(actions[right] - actions[left]))
            state_deltas.append(state_delta)
            action_deltas.append(action_delta)
            same = labels[left] == labels[right]
            (same_label_actions if same else changed_label_actions).append(action_delta)
            if state_delta > 1e-12:
                slope = action_delta / state_delta
                local_slopes.append(slope)
                (same_label_slopes if same else changed_label_slopes).append(slope)
    return {
        "transition_count": len(action_deltas),
        "state_delta": _quantiles(_finite(state_deltas)),
        "action_delta": _quantiles(_finite(action_deltas)),
        "same_label_action_delta": _quantiles(_finite(same_label_actions)),
        "changed_label_action_delta": _quantiles(_finite(changed_label_actions)),
        "action_state_local_slope": _quantiles(_finite(local_slopes)),
        "same_label_action_state_local_slope": _quantiles(_finite(same_label_slopes)),
        "changed_label_action_state_local_slope": _quantiles(_finite(changed_label_slopes)),
    }


def _label_durations(labels: np.ndarray, trajectory_ids: np.ndarray) -> dict[str, Any]:
    durations: list[int] = []
    switch_count = 0
    for trajectory in np.unique(trajectory_ids):
        indices = np.flatnonzero(trajectory_ids == trajectory)
        if indices.size == 0:
            continue
        current = int(labels[indices[0]])
        length = 1
        for index in indices[1:]:
            value = int(labels[index])
            if value == current:
                length += 1
            else:
                durations.append(length)
                switch_count += 1
                current = value
                length = 1
        durations.append(length)
    return {
        "trajectory_label_switch_count": switch_count,
        "segment_count": len(durations),
        "segment_duration": _quantiles([float(value) for value in durations]),
    }


def _label_report(data: dict[str, Any], labels: np.ndarray, name: str) -> dict[str, Any]:
    return {
        "label": name,
        "unique_labels": sorted(int(value) for value in np.unique(labels)),
        "action_relationship": _action_means_diagnostic(
            data["actions"], labels, data["splits"]
        ),
        "local_geometry": _transition_data(
            data["states"], data["actions"], labels, data["traj_ids"]
        ),
        "durations": _label_durations(labels, data["traj_ids"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["Can", "Square"])
    parser.add_argument("--cache-root", type=Path, default=REPO / "data/processed/cache")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs_cpu_debug/square_data_geometry_audit.json")
    )
    args = parser.parse_args(argv)
    report: dict[str, Any] = {
        "analysis": {
            "name": "square_data_geometry_audit",
            "cpu_only": True,
            "repository": str(REPO),
            "cache_root": str(args.cache_root.expanduser().resolve()),
            "tasks": list(args.tasks),
        },
        "tasks": {},
        "limitations": [
            "This uses demonstration-cache geometry and not closed-loop rollout states.",
            (
                "A large action/state local slope is an action-regression diagnostic, "
                "not a contact-force measurement."
            ),
            (
                "Within-label action variance can be necessary for fine control; it is "
                "not automatically harmful."
            ),
            (
                "Cache provenance and alignment must be checked before comparing phase "
                "and topology labels."
            ),
        ],
    }
    cache_root = args.cache_root.expanduser().resolve()
    try:
        for task in args.tasks:
            candidates = _candidate_caches(cache_root, task)
            phase_cache = _choose_cache(candidates, "phase")
            data = _load_cache(phase_cache)
            task_report: dict[str, Any] = {
                "phase_cache": str(phase_cache.path),
                "samples": int(len(data["phase"])),
                "trajectories": int(data["trajectory_count"]),
                "phase": _label_report(data, data["phase"], "phase"),
            }
            topo_candidates = [
                candidate for candidate in candidates if "phase_topo" in candidate.labels
            ]
            if topo_candidates:
                topo_cache = _choose_cache(topo_candidates, "phase_topo")
                topo_data = _load_cache(topo_cache)
                aligned, alignment_reason = _cache_alignment(data, topo_data)
                if not aligned:
                    task_report["topology"] = {
                        "available": False,
                        "reason": alignment_reason,
                    }
                else:
                    task_report["topology"] = {
                        "available": True,
                        "cache": str(topo_cache.path),
                        "alignment": alignment_reason,
                        **_label_report(data, topo_data["topo"], "phase_topo"),
                    }
            else:
                task_report["topology"] = {
                    "available": False,
                    "reason": "no aligned phase_topo cache found",
                }
            report["tasks"][task] = task_report
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        print(f"square_data_geometry_audit ERROR: {exc}", file=sys.stderr)
        return 2

    destination = args.out.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for task, result in report["tasks"].items():
        phase = result["phase"]["local_geometry"]
        print(
            f"{task}: phase_same_action_q50={phase['same_label_action_delta']['q50']} "
            f"phase_changed_action_q50={phase['changed_label_action_delta']['q50']}"
        )
    print(f"[square-data-geometry-audit] report={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
