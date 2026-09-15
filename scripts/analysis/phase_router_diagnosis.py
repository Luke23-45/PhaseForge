"""CPU-only diagnosis of phase observability versus router performance.

This diagnostic is intentionally separate from rollout evaluation.  It asks
whether the labels used to organize the latent/router are (a) inferable from
the instantaneous state and (b) useful for explaining action variation.  It
also reads archived Stage 1 summaries when they are available and reports
whether the phase-classification loss was actually active.

The distinction matters for the PhaseForge interpretation:

* a stable router cannot compensate for a representation that does not expose
  the relevant state distinctions;
* phase labels can be action-relevant while still being difficult to infer
  from one observation;
* a logged phase-head accuracy is not evidence of phase-head learning when
  SupCon's ``zero_ce`` contract suppresses the CE loss.

The script uses trajectory-grouped logistic-probe observability from the
repository implementation, train-fitted phase action means evaluated on the
held-out split, optional phase/topology label agreement, and archived Stage 1
metadata.  It never loads a GPU checkpoint and does not modify model or data
files.

Example::

    uv run python scripts/analysis/phase_router_diagnosis.py \
        --tasks Can Square \
        --outputs <focused-output-directory> \
        --out outputs_cpu_debug/phase_router_diagnosis.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TASK_ENV_NAMES = {
    "Lift": {"Lift"},
    "Can": {"Can", "PickPlaceCan"},
    "Square": {"Square", "NutAssemblySquare"},
    "ToolHang": {"ToolHang"},
    "Transport": {"Transport", "TwoArmTransport"},
}
TASK_RAW_DIR = {
    "Lift": "lift",
    "Can": "can",
    "Square": "square",
    "ToolHang": "tool_hang",
    "Transport": "transport",
}
NUM_PHASES = 6


@dataclass(frozen=True)
class CacheCandidate:
    task: str
    path: Path
    manifest: dict[str, Any]
    labels: frozenset[str]
    clean_status: str
    created_at: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _windows_long_path(path: Path) -> Path:
    """Return a Windows extended path when needed for deep archives."""
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return Path("\\\\?\\" + resolved)
    return Path(resolved)


def _manifest_configuration(manifest: dict[str, Any]) -> str:
    return str((manifest.get("provenance") or {}).get("configuration", ""))


def _setting(configuration: str, key: str) -> str | None:
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([^#\n]+)", configuration)
    return match.group(1).strip().lower() if match else None


def _clean_status(manifest: dict[str, Any]) -> str:
    configuration = _manifest_configuration(manifest)
    corruption = _setting(configuration, "phase_corruption_rate")
    shuffled = _setting(configuration, "phase_shuffle_control")
    if corruption is not None and float(corruption) != 0.0:
        return "nonclean"
    if shuffled in {"true", "yes", "1"}:
        return "nonclean"
    if corruption is None or shuffled is None:
        return "unspecified"
    return "declared_clean"


def _task_for_manifest(manifest: dict[str, Any]) -> str | None:
    metadata = (manifest.get("provenance") or {}).get("environment_metadata") or []
    if not metadata or not isinstance(metadata[0], dict):
        return None
    env_name = str(metadata[0].get("env_name", ""))
    for task, aliases in TASK_ENV_NAMES.items():
        if env_name in aliases:
            return task
    return None


def _candidate_caches(cache_root: Path, task: str) -> list[CacheCandidate]:
    candidates: list[CacheCandidate] = []
    for manifest_path in sorted(cache_root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("complete") is not True:
            continue
        if _task_for_manifest(manifest) != task:
            continue
        sampling = (manifest.get("provenance") or {}).get("sampling") or {}
        if sampling.get("sequence_length") != 1 or sampling.get("stride") != 1:
            continue
        trajectory_files = sorted((manifest_path.parent / "trajectories").glob("*.pt"))
        if not trajectory_files:
            continue
        try:
            first = torch.load(trajectory_files[0], map_location="cpu", weights_only=False)
        except Exception:
            continue
        labels = frozenset(str(k) for k in first if str(k).startswith("phase"))
        candidates.append(
            CacheCandidate(
                task=task,
                path=manifest_path.parent,
                manifest=manifest,
                labels=labels,
                clean_status=_clean_status(manifest),
                created_at=float(manifest.get("created_at", 0.0)),
            )
        )
    return candidates


def _choose_cache(
    candidates: list[CacheCandidate], required_label: str = "phase"
) -> CacheCandidate:
    usable = [c for c in candidates if required_label in c.labels and c.clean_status != "nonclean"]
    if not usable:
        raise FileNotFoundError(f"No usable sequence-length-1 cache contains {required_label!r}.")
    # Prefer explicitly clean caches, then the newest cache.  The choice is
    # recorded in the output so it is not silently treated as ground truth.
    return max(
        usable,
        key=lambda c: (c.clean_status == "declared_clean", c.created_at, c.path.name),
    )


def _load_cache(cache: CacheCandidate) -> dict[str, Any]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    topo: list[np.ndarray] = []
    traj_ids: list[np.ndarray] = []
    splits: list[str] = []
    demo_keys: list[str] = []
    topo_available = True

    files = sorted((cache.path / "trajectories").glob("*.pt"))
    for index, path in enumerate(files):
        item = torch.load(path, map_location="cpu", weights_only=False)
        state = np.asarray(item["state"], dtype=np.float64)
        action = np.asarray(item["action"], dtype=np.float64)
        phase = np.asarray(item["phase"], dtype=np.int64).reshape(-1)
        if (
            state.ndim != 2
            or action.ndim != 2
            or len(state) != len(action)
            or len(state) != len(phase)
        ):
            raise ValueError(f"Malformed trajectory {path}")
        states.append(state)
        actions.append(action)
        phases.append(phase)
        if "phase_topo" in item:
            topo.append(np.asarray(item["phase_topo"], dtype=np.int64).reshape(-1))
        else:
            topo_available = False
        traj_ids.append(np.full(len(state), index, dtype=np.int64))
        splits.extend([str(item["dataset_split"])] * len(state))
        demo_keys.extend([str(item["demo_key"])] * len(state))

    result: dict[str, Any] = {
        "states": np.concatenate(states),
        "actions": np.concatenate(actions),
        "phase": np.concatenate(phases),
        "traj_ids": np.concatenate(traj_ids),
        "splits": np.asarray(splits),
        "demo_keys": np.asarray(demo_keys),
        "topo": np.concatenate(topo) if topo_available and topo else None,
        "trajectory_count": len(files),
    }
    if result["topo"] is None:
        result["topo"] = None
    return result


def _action_means_diagnostic(
    actions: np.ndarray, labels: np.ndarray, splits: np.ndarray
) -> dict[str, Any]:
    train = splits == "train"
    val = splits == "val"
    if not train.any() or not val.any():
        raise ValueError("Cache must contain both train and validation samples.")
    global_mean = actions[train].mean(axis=0)
    global_mse = float(np.mean(np.square(actions[val] - global_mean)))
    means = np.tile(global_mean, (NUM_PHASES, 1))
    counts = np.bincount(labels[train], minlength=NUM_PHASES)
    for phase_id in range(NUM_PHASES):
        if counts[phase_id] > 0:
            means[phase_id] = actions[train & (labels == phase_id)].mean(axis=0)
    phase_mse = float(np.mean(np.square(actions[val] - means[labels[val]])))
    all_global = float(np.mean(np.square(actions - actions.mean(axis=0))))
    all_within = 0.0
    for phase_id in range(NUM_PHASES):
        mask = labels == phase_id
        if mask.any():
            within = float(np.mean(np.square(actions[mask] - actions[mask].mean(axis=0))))
            all_within += within * float(np.sum(mask))
    all_within /= float(len(actions))
    return {
        "heldout_global_action_mse": global_mse,
        "heldout_phase_mean_action_mse": phase_mse,
        "heldout_relative_mse_reduction": (
            1.0 - phase_mse / global_mse if global_mse > 0 else 0.0
        ),
        "all_data_phase_variance_explained": (
            1.0 - all_within / all_global if all_global > 0 else 0.0
        ),
        "train_phase_counts": counts.astype(int).tolist(),
    }


def _observability(
    states: np.ndarray,
    labels: np.ndarray,
    traj_ids: np.ndarray,
    actions: np.ndarray,
) -> dict[str, Any]:
    from phaseforge.data.topo.observability import audit_regimes

    return audit_regimes(
        states,
        labels,
        traj_ids,
        NUM_PHASES,
        actions=actions,
        min_macro_f1=0.6,
        min_occupancy=0.01,
    ).to_dict()


def _label_agreement(phase: np.ndarray, topo: np.ndarray) -> dict[str, float]:
    return {
        "normalized_mutual_information": float(normalized_mutual_info_score(phase, topo)),
        "adjusted_rand_index": float(adjusted_rand_score(phase, topo)),
    }


def _cache_alignment(first: dict[str, Any], second: dict[str, Any]) -> tuple[bool, str]:
    """Verify that two cache payloads refer to the same ordered samples."""
    if len(first["states"]) != len(second["states"]):
        return False, "sample counts differ"
    if not np.array_equal(first["demo_keys"], second["demo_keys"]):
        return False, "demo-key order differs"
    if not np.array_equal(first["splits"], second["splits"]):
        return False, "split order differs"
    if not np.allclose(first["states"], second["states"], rtol=0.0, atol=1e-8):
        return False, "state values differ"
    if not np.allclose(first["actions"], second["actions"], rtol=0.0, atol=1e-8):
        return False, "action values differ"
    return True, "state/action/demo-key order verified"


def _stage1_diagnostics(outputs: Path | None) -> dict[str, Any]:
    if outputs is None:
        return {"available": False, "reason": "Stage 1 output directory not supplied or not found."}
    outputs = _windows_long_path(outputs)
    if not outputs.exists():
        return {"available": False, "reason": "Stage 1 output directory not supplied or not found."}
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(outputs.glob("*/stage1/seed*/**/run_meta.json")):
        run_dir = meta_path.parent
        summary_path = run_dir / "metrics" / "summary.json"
        config_path = run_dir / "resolved_config.yaml"
        if not summary_path.is_file() or not config_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError):
            continue
        train_cfg = config.get("train", {}) or {}
        supcon = train_cfg.get("supcon", {}) or {}
        supcon_enabled = bool(supcon.get("enabled", False))
        zero_ce = bool(supcon.get("zero_ce", True))
        effective_ce = (
            0.0
            if supcon_enabled and zero_ce
            else float(train_cfg.get("lambda_phase", 1.0))
        )
        final = summary.get("final_val", {}) or {}
        rows.append(
            {
                "task": str(meta.get("tag", "")),
                "seed": int(meta.get("seed", -1)),
                "run_id": str(meta.get("run_id") or run_dir.name),
                "supcon_enabled": supcon_enabled,
                "supcon_zero_ce_effective_default": zero_ce,
                "effective_phase_ce_weight": effective_ce,
                "val_phase_acc": final.get("val/phase_acc"),
                "val_phase_balanced_acc": final.get("val/phase_balanced_acc"),
                "val_loss_phase": final.get("loss_phase"),
                "val_loss_action": final.get("loss_action"),
                "val_loss_supcon": final.get("loss_supcon"),
            }
        )
    if not rows:
        return {"available": False, "reason": "No complete Stage 1 summaries found."}
    return {"available": True, "rows": rows}


def _cache_metadata(cache: CacheCandidate, data_root: Path) -> dict[str, Any]:
    provenance = cache.manifest.get("provenance") or {}
    raw_files = provenance.get("raw_files") or []
    raw_dir = TASK_RAW_DIR[cache.task]
    raw_present = any(
        candidate.is_file()
        for item in raw_files
        for candidate in (
            data_root / "raw" / "robomimic" / raw_dir / str(item.get("name", "")),
            data_root / "raw" / raw_dir / str(item.get("name", "")),
            cache.path / str(item.get("name", "")),
        )
    )
    return {
        "path": str(cache.path),
        "cache_config_hash": cache.path.name,
        "created_at": cache.created_at,
        "clean_status": cache.clean_status,
        "git_commit": provenance.get("git_commit"),
        "raw_sha256_manifest": raw_files[0].get("sha256") if raw_files else None,
        "raw_files_present": raw_present,
        "manifest_sha256": _sha256(cache.path / "manifest.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["Can", "Square"])
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--outputs", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs_cpu_debug/phase_router_diagnosis.json"),
    )
    args = parser.parse_args(argv)

    data_root = _windows_long_path(args.data_root)
    cache_root = _windows_long_path(args.cache_root or data_root / "processed" / "cache")
    report: dict[str, Any] = {
        "analysis": {
            "name": "phase_router_diagnosis",
            "cpu_only": True,
            "repository": str(REPO),
            "tasks": list(args.tasks),
            "data_root": str(data_root),
            "cache_root": str(cache_root),
        "outputs_root": str(_windows_long_path(args.outputs)) if args.outputs else None,
            "observability_threshold_macro_f1": 0.6,
            "num_phases": NUM_PHASES,
        },
        "tasks": {},
        "stage1": _stage1_diagnostics(
            args.outputs.expanduser().resolve() if args.outputs else None
        ),
        "limitations": [
            "This is a state/data diagnostic, not a replacement for closed-loop "
            "rollout evaluation.",
            "Raw HDF5 files are not required by this analysis and are reported "
            "as unverified when absent.",
            "The local cache may not be the exact cloud cache used by the "
            "archived focused run; cache provenance is retained above.",
            "A missing phase_topo cache is reported as missing rather than inferred or fabricated.",
        ],
    }

    try:
        for task in args.tasks:
            candidates = _candidate_caches(cache_root, task)
            phase_cache = _choose_cache(candidates, "phase")
            data = _load_cache(phase_cache)
            task_report: dict[str, Any] = {
                "phase_cache": _cache_metadata(phase_cache, data_root),
                "samples": int(len(data["phase"])),
                "trajectories": int(data["trajectory_count"]),
                "phase_observability": _observability(
                    data["states"], data["phase"], data["traj_ids"], data["actions"]
                ),
                "phase_action_relationship": _action_means_diagnostic(
                    data["actions"], data["phase"], data["splits"]
                ),
            }
            topo_candidates = [c for c in candidates if "phase_topo" in c.labels]
            if topo_candidates:
                topo_cache = _choose_cache(topo_candidates, "phase_topo")
                topo_data = _load_cache(topo_cache)
                task_report["topology_cache"] = _cache_metadata(topo_cache, data_root)
                task_report["phase_topology_observability"] = _observability(
                    topo_data["states"],
                    topo_data["topo"],
                    topo_data["traj_ids"],
                    topo_data["actions"],
                )
                aligned, alignment_reason = _cache_alignment(data, topo_data)
                task_report["phase_topology_alignment"] = {
                    "verified": aligned,
                    "reason": alignment_reason,
                }
                if aligned:
                    task_report["phase_topology_label_agreement"] = _label_agreement(
                        data["phase"], topo_data["topo"]
                    )
                else:
                    task_report["phase_topology_label_agreement"] = None
            else:
                task_report["topology_cache"] = None
                task_report["phase_topology_observability"] = None
                task_report["phase_topology_alignment"] = None
                task_report["phase_topology_label_agreement"] = None
            report["tasks"][task] = task_report
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        print(f"phase_router_diagnosis ERROR: {exc}", file=sys.stderr)
        return 2

    output_path = args.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[phase-router-diagnosis] report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
