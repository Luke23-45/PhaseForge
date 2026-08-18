"""Collect GPU re-run eval + training metrics and print a comparison table."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("outputs")
PARTS = ["part1", "part2"]


def load_rollouts() -> dict:
    rows = {}
    for part in PARTS:
        eval_root = ROOT / part / "outputs" / "eval"
        if not eval_root.is_dir():
            continue
        for f in eval_root.rglob("rollout_summary.json"):
            p = f.parent
            seed = p.parent.name
            if seed.startswith("seed"):
                seed = seed[4:]
            method = p.parent.parent.name
            j = json.loads(f.read_text())
            m = j.get("metrics", {})
            row = {
                "success": m.get("eval/rollout/success_rate"),
                "n": m.get("eval/rollout/valid_episodes"),
                "successes": m.get("eval/rollout/successes"),
                "timeouts": j.get("failure_categories", {}).get("task_timeout"),
                "pfail": m.get("eval/rollout/policy_failures"),
                "invalid": m.get("eval/rollout/invalid_attempts"),
                "ci_low": m.get("eval/rollout/wilson_ci95_low"),
                "ci_high": m.get("eval/rollout/wilson_ci95_high"),
                "reset_bank": m.get("eval/rollout/reset_bank") or j.get("reset_bank"),
                "ckpt_sha": j.get("checkpoint_sha256"),
            }
            rows[(method, seed)] = row
    return rows


def load_training_summaries() -> dict:
    rows = {}
    for part in PARTS:
        out_root = ROOT / part / "outputs"
        for f in out_root.rglob("metrics/summary.json"):
            rel = f.relative_to(out_root).parts
            # phaseforge/stage1/seed42/<run>/metrics/summary.json
            if len(rel) < 4:
                continue
            method, stage, seed = rel[0], rel[1], rel[2]
            if not stage.startswith("stage"):
                continue
            if seed.startswith("seed"):
                seed = seed[4:]
            rows[(method, stage, seed)] = json.loads(f.read_text())
    return rows


def main() -> None:
    rollouts = load_rollouts()
    trains = load_training_summaries()

    print("=== ROLLOUT SUCCESS (GPU re-run @ c09270a) ===")
    methods = sorted({m for m, _ in rollouts})
    for method in methods:
        vals = []
        for seed in ("42", "43", "44"):
            j = rollouts.get((method, seed))
            if j is None:
                vals.append("   -  ")
                continue
            v = j.get("success")
            vals.append(f"{v:.3f}")
            print(
                f"{method:>35} seed{seed}: success={v:.3f} n={j.get('n')} "
                f"tmo={j.get('timeouts')} pfail={j.get('pfail')} invalid={j.get('invalid')} "
                f"CI=[{j.get('ci_low'):.3f},{j.get('ci_high'):.3f}]"
            )
        print(f"{'':>35} pooled: {' '.join(vals)}")

    print()
    print("=== STAGE-2 TRAINING (final val metrics) ===")
    for method in sorted({m for m, _, _ in trains}):
        for stage in ("stage1", "stage2"):
            rows = [trains.get((method, stage, s)) for s in ("42", "43", "44")]
            if not any(rows):
                continue
            print(f"--- {method} {stage} ---")
            for seed, r in zip(("42", "43", "44"), rows):
                if r is None:
                    continue
                fv = r.get("final_val", {})
                keys = {
                    k: fv.get(k)
                    for k in ("loss_action", "loss_phase", "phase_expert_nmi", "top1_collapse_rate", "top1_balance_score")
                    if fv.get(k) is not None
                }
                print(
                    f"    seed{seed}: best_ep={r.get('best_epoch')} "
                    f"best_mon={r.get('best_val_monitor'):.4f} "
                    + " ".join(f"{k}={v:.4f}" for k, v in keys.items())
                )


if __name__ == "__main__":
    main()