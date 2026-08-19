"""Gate G2: phase-merge / separability check (offline, no training).

Decides V2-A's scope — soft labels vs a simpler merged phase
decomposition — and feeds the M construction decision (report1.md §G2).

Measures how separable the six causal phases are in the *normalized state
space* the phase head actually classifies (frozen train-split normalizer,
same processed cache the pipeline trains on):

* per-phase support size and intra-phase dispersion;
* pairwise centroid separation d' (Cohen's d) and centroid-classifier
  confusion ``conf(p -> q)`` = fraction of phase-p states nearer to
  phase-q's centroid than to their own — the number that drives the
  phase-head confusion at train time;
* the merged schemes: 6 phases (current), 5 (merge grasp+transport
  ``{2,3}``), and 4 super-phases (``{0,1}`` approach, ``{2,3}``
  grasp/transport, ``{4}`` place, ``{5}`` retract), each with mean
  pairwise confusion and a closed-form silhouette.

The gate verdict (printed, also in the JSON): whether the 2<->3 (and
1<->5) confusions are high enough that merging removes more confusion
than the phase vocabulary loses.

Outputs:
    outputs/_findings/phase_merge_separability.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FINDINGS_DIR = Path("outputs/_findings")

PHASE_NAMES = {
    0: "approach",
    1: "pre-grasp",
    2: "grasp",
    3: "transport",
    4: "place",
    5: "retract",
}

#: Candidate merged decompositions, as super-phase -> member phases.
MERGED_SCHEMES: dict[str, tuple[int, ...]] = {
    "6_phases": (0, 1, 2, 3, 4, 5),
    "5_merge_grasp_transport": (0, 1, 2, 3, 4, 5),  # placeholder, built below
    "4_super": (0, 1, 2, 3, 4, 5),  # placeholder, built below
}


def _load_cache(cfg, cache_dir: Path | None = None) -> tuple[list[dict], dict, dict, Path]:
    from phaseforge.data.ingestion.cache_manager import CacheManager
    from phaseforge.data.paths import processed_cache_root

    if cache_dir is None:
        hash_val = CacheManager.compute_hash(cfg.data)
        cache_dir = processed_cache_root() / hash_val
    cache_dir = Path(cache_dir)
    if not (cache_dir / "trajectories").is_dir():
        raise FileNotFoundError(
            f"processed cache missing under {cache_dir} — ingest the raw "
            "HDF5 before running gate G2 (or pass --cache-dir)"
        )
    manager = CacheManager(processed_cache_root())
    trajectories, norm_stats, splits, _task_index = manager.load(cache_dir.name)
    return trajectories, norm_stats, splits, cache_dir


def _split_states(
    trajectories: list[dict], norm_stats: dict, splits: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized train-split state matrix + phase labels."""
    train_idx = set(splits.get("train", []))
    mean = np.asarray(norm_stats["mean"], dtype=np.float64)
    std = np.asarray(norm_stats["std"], dtype=np.float64)
    std = np.where(std == 0.0, 1.0, std)
    xs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for i, traj in enumerate(trajectories):
        if train_idx and i not in train_idx:
            continue
        state = np.asarray(traj["state"], dtype=np.float64)
        xs.append((state - mean) / std)
        labels.append(np.asarray(traj["phase"], dtype=np.int64))
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def _by_phase(X: np.ndarray, y: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for p in np.unique(y):
        out[int(p)] = X[y == p]
    return out


def _centroid_confusions(groups: dict[int, np.ndarray]) -> dict[str, float]:
    """Centroid-classifier confusion between every pair of groups.

    ``conf(p -> q)`` = fraction of group-p states nearer to group-q's
    centroid than to group-p's own centroid (0 = perfectly separable,
    1 = fully confusable). The mean over ordered pairs is the overall
    confusion of the decomposition.
    """
    centroids = {p: xs.mean(axis=0) for p, xs in groups.items()}
    confusions: dict[str, float] = {}
    for p, xs in groups.items():
        own = np.linalg.norm(xs - centroids[p], axis=1)
        for q in groups:
            if q == p:
                continue
            other = np.linalg.norm(xs - centroids[q], axis=1)
            confusions[f"{p}->{q}"] = float((other < own).mean())
    return confusions


def _cohens_d(groups: dict[int, np.ndarray]) -> dict[str, float]:
    ds: dict[str, float] = {}
    keys = sorted(groups)
    for i, p in enumerate(keys):
        for q in keys[i + 1 :]:
            c_p, c_q = groups[p].mean(axis=0), groups[q].mean(axis=0)
            s_p, s_q = groups[p].std(axis=0).mean(), groups[q].std(axis=0).mean()
            s_pooled = np.sqrt((s_p**2 + s_q**2) / 2) + 1e-12
            ds[f"{p}|{q}"] = float(np.linalg.norm(c_p - c_q) / s_pooled)
    return ds


def _silhouette(groups: dict[int, np.ndarray], max_samples: int) -> float | None:
    """Closed-form silhouette: (b - a) / max(a, b) averaged per group."""
    if len(groups) < 2:
        return None
    centroids = {p: xs.mean(axis=0) for p, xs in groups.items()}
    scores: list[float] = []
    for p, xs in groups.items():
        xs = xs[:max_samples]
        a = np.linalg.norm(xs - centroids[p], axis=1).mean()
        others = [q for q in groups if q != p]
        b = min(np.linalg.norm(xs - centroids[o], axis=1).mean() for o in others)
        scores.append(float((b - a) / max(a, b, 1e-12)))
    return float(np.mean(scores))


def _merge(groups: dict[int, np.ndarray], members: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Rebuild groups under a super-phase assignment.

    ``members`` maps each phase 0..P-1 to its super-phase index; super-phases
    are numbered by first appearance.
    """
    merged: dict[int, list[np.ndarray]] = {}
    super_of: dict[int, int] = {}
    next_id = 0
    for phase in sorted(groups):
        super_id = members[phase]
        if super_id not in super_of:
            super_of[super_id] = next_id
            next_id += 1
        merged.setdefault(super_of[super_id], []).append(groups[phase])
    return {k: np.concatenate(v, axis=0) for k, v in sorted(merged.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="common",
        help="Data config group whose processed cache is analyzed. Default "
        "'common' matches the lift_ablation manifest cells (data=common); "
        "use --cache-dir to point at any specific cache.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Use a specific processed-cache directory instead of resolving the hash from --data",
    )
    parser.add_argument("--max-samples", type=int, default=200_000)
    args = parser.parse_args(argv)

    from hydra import initialize_config_dir

    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "phaseforge/config"),
        version_base=None,
        job_name="gate_g2",
    ):
        from hydra import compose

        cfg = compose(config_name="main", overrides=[f"data={args.data}"])

    trajectories, norm_stats, splits, cache_dir = _load_cache(cfg, cache_dir=args.cache_dir)
    X, y = _split_states(trajectories, norm_stats, splits)
    print(f"[g2] cache: {len(trajectories)} trajectories, {X.shape[0]} train states")
    groups = _by_phase(X, y)

    per_phase = {}
    for p in sorted(groups):
        xs = groups[p]
        per_phase[str(p)] = {
            "name": PHASE_NAMES.get(p, "?"),
            "n": int(xs.shape[0]),
            "intra_dispersion": float(xs.std(axis=0).mean()),
        }

    confusions = _centroid_confusions(groups)
    dprime = _cohens_d(groups)
    silhouette_6 = _silhouette(groups, args.max_samples)

    schemes: dict[str, dict] = {}
    schemes["6_phases"] = {
        "members": "0 1 2 3 4 5",
        "mean_confusion": float(np.mean(list(confusions.values()))),
        "silhouette": silhouette_6,
    }
    # 5 phases: merge grasp (2) + transport (3).
    groups_5 = _merge(groups, (0, 1, 2, 2, 4, 5))
    conf_5 = _centroid_confusions(groups_5)
    # In the merged space the {2,3} group is index 2; its neighbors are the
    # pre-grasp group (1) and the place group (3). Residual confusion against
    # the neighbors is what the merged group still has to pay.
    residual_23 = max(
        conf_5.get("2->1", 0.0),
        conf_5.get("1->2", 0.0),
        conf_5.get("2->3", 0.0),
        conf_5.get("3->2", 0.0),
    )
    schemes["5_merge_grasp_transport"] = {
        "members": "0 1 {2,3} 4 5",
        "mean_confusion": float(np.mean(list(conf_5.values()))),
        "silhouette": _silhouette(groups_5, args.max_samples),
        "residual_neighbor_confusion": residual_23,
    }
    # 4 super-phases: approach {0,1}, grasp/transport {2,3}, place {4}, retract {5}.
    groups_4 = _merge(groups, (0, 0, 1, 1, 2, 3))
    conf_4 = _centroid_confusions(groups_4)
    schemes["4_super"] = {
        "members": "{0,1} {2,3} 4 5",
        "mean_confusion": float(np.mean(list(conf_4.values()))),
        "silhouette": _silhouette(groups_4, args.max_samples),
    }

    # Gate verdict: the pairwise 2<->3 (and 1<->5) confusions decide whether
    # merging removes more confusion than the vocabulary loses.
    conf_23 = max(confusions.get("2->3", 0.0), confusions.get("3->2", 0.0))
    conf_15 = max(confusions.get("1->5", 0.0), confusions.get("5->1", 0.0))
    verdict = {
        "merge_2_3_recommended": bool(
            conf_23 > 0.25
            and schemes["5_merge_grasp_transport"]["silhouette"]
            >= schemes["6_phases"]["silhouette"]
        ),
        "conf_2_3": conf_23,
        "conf_1_5": conf_15,
        "residual_neighbor_confusion_after_merge_2_3": residual_23,
    }

    rel_cache = (
        cache_dir.relative_to(PROJECT_ROOT) if cache_dir.is_relative_to(PROJECT_ROOT) else cache_dir
    )
    payload = {
        "data": args.data,
        "cache_dir": str(rel_cache),
        "n_train_states": int(X.shape[0]),
        "per_phase": per_phase,
        "pairwise": {
            "confusions": confusions,
            "cohens_d": dprime,
        },
        "schemes": schemes,
        "verdict": verdict,
    }
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    (FINDINGS_DIR / "phase_merge_separability.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("\n[g2] per-phase support:")
    for key, info in per_phase.items():
        print(
            f"  phase {p} ({info['name']}): n={info['n']}, "
            f"dispersion={info['intra_dispersion']:.4f}"
        )
    print("\n[g2] pairwise confusions (centroid classifier):")
    for pair in sorted(confusions):
        print(f"  {pair}: {confusions[pair]:.3f}")
    print("\n[g2] Cohen's d (centroid separation):")
    for pair in sorted(dprime):
        print(f"  {pair}: {dprime[pair]:.3f}")
    print("\n[g2] merged schemes:")
    for name, info in schemes.items():
        print(
            f"  {name}: mean_confusion={info['mean_confusion']:.3f}, "
            f"silhouette={info['silhouette']:.3f}"
        )
    print(
        f"\n[g2] VERDICT: merge 2+3 recommended = {verdict['merge_2_3_recommended']} "
        f"(conf 2<->3 = {conf_23:.3f}, conf 1<->5 = {conf_15:.3f})"
    )
    print(f"[g2] written to {FINDINGS_DIR / 'phase_merge_separability.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
