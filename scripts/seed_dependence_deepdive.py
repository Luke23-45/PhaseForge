"""Deep-dive: why is PhaseForge rollout success seed-dependent (s44=0.50 vs 0.68/0.74)?

Tests hypotheses against actual artifacts:
  H1: frozen stage-1 encoder quality varies per seed (phase-head / action-head)
  H2: router/phase structure varies per seed (NMI, entropy, balance, collapse)
  H3: eval checkpoint is NOT the best-monitor epoch (wrong checkpoint picked per seed)
  H4: per-episode failure pattern is boundary noise (near-threshold episodes flip)
"""

from __future__ import annotations

import json
from pathlib import Path

PARTS = ["part1", "part2"]
METHOD = "phaseforge"
SEEDS = ["42", "43", "44"]


def find_run(method: str, seed: str, stage: str, part: str) -> Path | None:
    d = Path("outputs") / part / "outputs" / method / f"stage{stage}" / f"seed{seed}"
    if not d.is_dir():
        return None
    runs = sorted(d.iterdir())
    return runs[0] if runs else None


def load_metrics_file(run: Path) -> dict:
    p = run / "metrics" / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_history(run: Path) -> list[dict]:
    p = run / "metrics" / "training_curves.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def load_episodes(method: str, seed: str) -> list[dict]:
    for part in PARTS:
        d = Path("outputs") / part / "outputs" / "eval" / method / f"seed{seed}"
        if d.is_dir():
            run = sorted(d.iterdir())[0]
            return [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines()]
    raise FileNotFoundError(f"{method} seed{seed}")


def main() -> None:
    print("=== H1/H2: stage-1 final + phase structure per seed ===")
    for seed in SEEDS:
        run = find_run(METHOD, seed, 1, "part1") or find_run(METHOD, seed, 1, "part2")
        j = load_metrics_file(run)
        fv = j.get("final_val", {})
        keys = ("loss_action", "loss_phase", "val/phase_head_acc", "val/phase_expert_nmi",
                "val/routing_entropy", "val/top1_balance_score", "val/top1_collapse_rate")
        print(f"  seed{seed}: best_ep={j.get('best_epoch')} best_mon={j.get('best_val_monitor'):.4f}")
        for k in keys:
            if k in fv:
                print(f"      {k} = {fv[k]:.4f}" if isinstance(fv[k], float) else f"      {k} = {fv[k]}")
        hist = load_history(run)
        if hist:
            last = hist[-1]
            print(f"      last_epoch={last.get('epoch')} loss_action={last.get('val/loss_action', '?'):.4f}")

    print()
    print("=== H1b: stage-1 phase-head accuracy over epochs (s42 vs s44) ===")
    for seed in SEEDS:
        run = find_run(METHOD, seed, 1, "part1") or find_run(METHOD, seed, 1, "part2")
        hist = load_history(run)
        if not hist:
            print(f"  seed{seed}: no history.jsonl")
            continue
        cols = ("epoch", "val/phase_head_acc", "val/loss_phase", "val/loss_action")
        print(f"  seed{seed}:")
        for h in hist[:: max(1, len(hist) // 10)]:
            print("     " + "  ".join(f"{c}={h.get(c, float('nan')):.4f}" for c in cols))

    print()
    print("=== H2: stage-2 final routing metrics per seed ===")
    for seed in SEEDS:
        run = find_run(METHOD, seed, 2, "part1") or find_run(METHOD, seed, 2, "part2")
        j = load_metrics_file(run)
        fv = j.get("final_val", {})
        print(f"  seed{seed}: best_ep={j.get('best_epoch')} best_mon={j.get('best_val_monitor'):.4f}")
        for k in ("val/phase_expert_nmi", "val/routing_entropy", "val/top1_balance_score",
                  "val/top1_collapse_rate", "loss_action"):
            if k in fv:
                print(f"      {k} = {fv[k]:.4f}")

    print()
    print("=== H3: eval checkpoint vs best-monitor epoch (identity) ===")
    for seed in SEEDS:
        run = find_run(METHOD, seed, 2, "part1") or find_run(METHOD, seed, 2, "part2")
        j = load_metrics_file(run)
        best_ep = j.get("best_epoch")
        ck_dir = run / "checkpoints"
        ckpts = sorted(ck_dir.iterdir()) if ck_dir.is_dir() else []
        print(f"  seed{seed}: best_ep={best_ep} checkpoints={[c.name for c in ckpts]}")

    print()
    print("=== H4: per-episode success pattern (all three seeds) ===")
    eps = {s: load_episodes(METHOD, s) for s in SEEDS}
    for s in SEEDS:
        assert len(eps[s]) == 50, f"seed {s}: {len(eps[s])} episodes"
    for i in range(50):
        pat = "".join("S" if eps[s][i]["success"] else "F" for s in SEEDS)
        if pat != "SSS":
            steps = [eps[s][i]["steps"] for s in SEEDS]
            print(f"  ep{i:>3}: {pat} steps={steps}")

    print()
    print("=== Episode-level: mean steps-to-success per seed (task difficulty proxy) ===")
    for s in SEEDS:
        succ_steps = [e["steps"] for e in eps[s] if e["success"]]
        print(f"  seed{s}: mean_steps_on_success={sum(succ_steps)/len(succ_steps):.0f} n={len(succ_steps)}")


    print()
    print("=== H5: phase-head quality at the SELECTED stage-1 checkpoint (the frozen encoder) ===")
    for seed in SEEDS:
        run = find_run(METHOD, seed, 1, "part1") or find_run(METHOD, seed, 1, "part2")
        hist = load_history(run)
        if not hist:
            continue
        best_ep = min(hist, key=lambda r: r.get("val/loss_action", 1e9))
        final = hist[-1]
        keys = ("val/loss_phase", "val/loss_action", "val/phase_head_acc")
        print(f"  seed{seed}: selected_ep={best_ep['epoch']} "
              + " ".join(f"{k.split('/')[-1]}@{best_ep['epoch']}={best_ep.get(k)}" for k in keys)
              + "  |  " + " ".join(f"{k.split('/')[-1]}_final={final.get(k)}" for k in keys))
        for r in hist:
            print(f"     ep{r['epoch']:>3}: phase={r.get('val/loss_phase', float('nan')):.4f} "
                  f"act={r.get('val/loss_action', float('nan')):.4f} "
                  f"acc={r.get('val/phase_head_acc', float('nan'))}")

    print()
    print("=== H6: does router init explain spread? (phaseforge=centroid vs phase_pretrain=random) ===")
    for m in ("phaseforge", "phase_pretrain_random_router"):
        vals = []
        for seed in SEEDS:
            run = find_run(m, seed, 2, "part1") or find_run(m, seed, 2, "part2")
            j = load_metrics_file(run)
            fv = j.get("final_val", {})
            vals.append((seed, j.get("best_epoch"), fv.get("val/phase_expert_nmi"), fv.get("loss_action")))
        print(f"  {m}: " + " ".join(f"s{s}:ep={e},nmi={n:.3f},act={a:.3f}" for s, e, n, a in vals))


if __name__ == "__main__":
    main()