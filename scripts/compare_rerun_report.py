"""Full comparison: GPU re-run (c09270a, fixed monitor) vs the old report numbers."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("outputs")
PARTS = ["part1", "part2"]

NMI_KEYS = ("val/phase_expert_nmi", "val/routing_entropy", "val/top1_balance_score", "val/top1_collapse_rate")
NMI_LABELS = {"val/phase_expert_nmi": "NMI", "val/routing_entropy": "entropy", "val/top1_balance_score": "top1_bal", "val/top1_collapse_rate": "collapse"}


def load_rollouts() -> dict:
    rows = {}
    for part in PARTS:
        for f in (ROOT / part / "outputs" / "eval").rglob("rollout_summary.json"):
            p = f.parent
            seed = p.parent.name
            seed = seed[4:] if seed.startswith("seed") else seed
            method = p.parent.parent.name
            j = json.loads(f.read_text())
            m = j.get("metrics", {})
            rows[(method, seed)] = {
                "success": m.get("eval/rollout/success_rate"),
                "ci_low": m.get("eval/rollout/wilson_ci95_low"),
                "ci_high": m.get("eval/rollout/wilson_ci95_high"),
                "timeouts": j.get("failure_categories", {}).get("task_timeout"),
            }
    return rows


def load_stage2() -> dict:
    rows = {}
    for part in PARTS:
        out_root = ROOT / part / "outputs"
        for f in out_root.rglob("stage2/seed*/*/metrics/summary.json"):
            rel = f.relative_to(out_root).parts
            method = rel[0]
            seed = rel[2][4:] if rel[2].startswith("seed") else rel[2]
            j = json.loads(f.read_text())
            fv = j.get("final_val", {})
            rows[(method, seed)] = {
                "best_ep": j.get("best_epoch"),
                "best_mon": j.get("best_val_monitor"),
                "loss_action": fv.get("loss_action"),
                **{k: fv.get(k) for k in NMI_KEYS},
            }
    return rows


def wilson_pooled(successes: list) -> tuple:
    """Wilson 95% CI over pooled episodes (each seed contributes n=50)."""
    k = sum(int(round(s * 50)) for s in successes)
    n = 50 * len(successes)
    p = k / n
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / denom
    return p, centre - half, centre + half


def main() -> None:
    rollouts = load_rollouts()
    stage2 = load_stage2()

    methods = ["phaseforge", "phase_pretrain_random_router", "plain_encoder_phase_bootstrap",
               "warmstart_moe", "scratch_moe", "bc"]
    labels = {
        "phaseforge": "PhaseForge (proposed)",
        "phase_pretrain_random_router": "Phase-Pretrain Random-Router",
        "plain_encoder_phase_bootstrap": "Plain-Enc. Phase-Bootstrap",
        "warmstart_moe": "Warm-Start MoE",
        "scratch_moe": "Scratch MoE",
        "bc": "BC-MLP (floor)",
    }
    # Old (buggy-monitor) rollout numbers from the report.
    old = {
        "phaseforge": [0.56, 0.72, 0.42],
        "phase_pretrain_random_router": [0.60, 0.50, 0.28],
        "plain_encoder_phase_bootstrap": [0.58, 0.62, 0.60],
        "warmstart_moe": [0.58, 0.56, 0.40],
        "scratch_moe": [0.58, 0.66, 0.52],
        "bc": [0.60, 0.48, 0.54],
    }

    print("ROLLOUT SUCCESS — FIXED GPU RE-RUN (c09270a) vs OLD REPORT (buggy monitor)")
    print(f"{'method':<34} {'s42':>5} {'s43':>5} {'s44':>5} {'mean':>6} {'spread':>6} | old mean  old spread")
    print("-" * 100)
    for m in methods:
        vals = [rollouts[(m, s)]["success"] for s in ("42", "43", "44")]
        pooled, lo, hi = wilson_pooled(vals)
        ov = old[m]
        print(f"{labels[m]:<34} "
              f"{vals[0]:5.2f} {vals[1]:5.2f} {vals[2]:5.2f} "
              f"{statistics.mean(vals):6.3f} {max(vals)-min(vals):6.2f} | "
              f"{statistics.mean(ov):6.3f}  {max(ov)-min(ov):6.2f}   CI[{lo:.3f},{hi:.3f}]")

    print()
    print("STAGE-2 ROUTING (final val) + rollout success — fixed run")
    print(f"{'method':<34} {'s42':>5} {'s43':>5} {'s44':>5}  NMI(42/43/44)   entropy     collapse")
    for m in methods:
        if m == "bc":
            continue
        r = stage2.get((m, "42"))
        if r is None:
            continue
        n = [stage2[(m, s)]["val/phase_expert_nmi"] for s in ("42", "43", "44")]
        en = [stage2[(m, s)]["val/routing_entropy"] for s in ("42", "43", "44")]
        co = [stage2[(m, s)]["val/top1_collapse_rate"] for s in ("42", "43", "44")]
        vals = [rollouts[(m, s)]["success"] for s in ("42", "43", "44")]
        print(f"{labels[m]:<34} {vals[0]:5.2f} {vals[1]:5.2f} {vals[2]:5.2f}  "
              f"{n[0]:.3f}/{n[1]:.3f}/{n[2]:.3f}   {statistics.mean(en):.3f}   {statistics.mean(co)*100:.0f}%")

    print()
    print("STAGE-1 BEST (fixed run) — the encoder every stage-2 freezes")
    for m in ("phaseforge", "bc"):
        r = stage2.get((m, "42"))
    for f in (ROOT / "part1" / "outputs" / "phaseforge" / "stage1").rglob("metrics/summary.json"):
        seed = f.parent.parent.name[4:]
        j = json.loads(f.read_text())
        print(f"  phaseforge stage1 seed{seed}: best_ep={j['best_epoch']:>3} best_val/action={j['best_val_monitor']:.4f}")


if __name__ == "__main__":
    main()