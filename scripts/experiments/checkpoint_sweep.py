"""Wave A1: checkpoint sweep — SR vs epoch on the same 50 paired episodes.

Trains phaseforge stage-1 and stage-2 (checkpoints every ``--ckpt-cadence``
epochs) for the given seeds, then rollout-evaluates the requested checkpoint
epochs on the frozen reset bank (seed 2026, 50 cases — identical episodes
across checkpoints). Overlays the training telemetry (val action loss, NMI,
routing entropy, balance, collapse) per epoch and reports SR-vs-val-loss
correlation.

Evidence-first probe defaults: a single seed (42) and the epoch set
{10, 30, 100, 200, best} — the span of training plus the protocol-selected
checkpoint. That is enough to decide whether checkpoint selection is a
bottleneck; expand seeds/epochs only if the signal warrants it. Stage-1
and stage-2 runs already present are reused; completed evals are reused
unless ``--force-evals``.

Outputs:
    outputs/surgical/_findings/checkpoint_sweep.json
    docs/dev/findings/checkpoint_sweep.md

Usage:
    python scripts/experiments/checkpoint_sweep.py [--seeds 42]
        [--stage1-epochs 100] [--epochs 200] [--ckpt-cadence 10]
        [--eval-epochs 10,30,100,200,best]
        [--outputs outputs/surgical] [--skip-provider] [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from phaseforge.runner.commands import eval_command, train_command
from phaseforge.runner.protocol import Step, load_protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from smoke_matrix import (  # noqa: E402
    EVAL_QUIET,
    STAGE2_QUIET,
    TRAIN_QUIET,
    _execute,
    _find_run_by_meta,
    _run_dir_base,
)

FINDINGS_DIR = Path("outputs/surgical/_findings")
REPORT_PATH = Path("docs/dev/findings/checkpoint_sweep.md")
EVAL_METHOD = "phaseforge_cpsweep"


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    if den == 0:
        return None
    return num / den


def _step(method, seed: int, stage: int | None) -> Step:
    return Step(kind="train" if stage is not None else "eval", method=method, seed=seed, stage=stage)


def _find_eval_run(outputs: Path, model_name: str, seed: int, tag: str) -> Path:
    base = outputs / "eval" / model_name / f"seed{seed}"
    return _find_run_by_meta(base, EVAL_METHOD, seed, tag)


def _train_stage1(provider, seed: int, outputs: Path, defaults, epochs: int, logs: Path, timeout: int) -> Path:
    quiet = list(TRAIN_QUIET) + [f"train.epochs={epochs}"]
    step = _step(provider, seed, 1)
    cmd = train_command(step, outputs_base=outputs, defaults=defaults + tuple(quiet), ckpt_path=None)
    _execute(["phaseforge-train", "project.log_level=WARNING"] + cmd[1:],
             logs / f"stage1_seed{seed}.log", timeout)
    run_dir = _find_run_by_meta(_run_dir_base(outputs, provider, 1, seed), provider.name, seed, None)
    ckpt = run_dir / "checkpoints" / "checkpoint_best.pt"
    if not ckpt.is_file():
        raise RuntimeError(f"no checkpoint_best.pt under {run_dir}")
    return run_dir


def _train_stage2(provider, seed: int, outputs: Path, defaults, epochs: int, cadence: int, ckpt_path: Path, logs: Path, timeout: int) -> Path:
    quiet = list(TRAIN_QUIET) + list(STAGE2_QUIET) + [
        f"train.epochs={epochs}",
        f"train.checkpoint.every_n_epochs={cadence}",
    ]
    step = _step(provider, seed, 2)
    cmd = train_command(step, outputs_base=outputs, defaults=defaults + tuple(quiet), ckpt_path=ckpt_path)
    _execute(["phaseforge-train", "project.log_level=WARNING"] + cmd[1:],
             logs / f"stage2_seed{seed}.log", timeout)
    return _find_run_by_meta(_run_dir_base(outputs, provider, 2, seed), provider.name, seed, None)


def _read_eval_summary(run_dir: Path) -> dict | None:
    summary = run_dir / "rollout_summary.json"
    if not summary.is_file():
        return None
    data = json.loads(summary.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    return {
        "sr": metrics.get("eval/rollout/success_rate"),
        "ci_low": metrics.get("eval/rollout/wilson_ci95_low"),
        "ci_high": metrics.get("eval/rollout/wilson_ci95_high"),
    }


def _eval_epoch(outputs: Path, provider, seed: int, ckpt_path: Path, epoch_label: str, defaults, logs: Path, timeout: int) -> dict:
    method = dataclasses.replace(
        provider, name=EVAL_METHOD, tag=f"cp_e{epoch_label}"
    )
    step = _step(method, seed, None)
    cmd = eval_command(step, ckpt_path=ckpt_path, outputs_base=outputs, defaults=defaults + EVAL_QUIET)
    _execute(["phaseforge-eval", "project.log_level=WARNING"] + cmd[1:],
             logs / f"eval_seed{seed}_{epoch_label}.log", timeout)
    run_dir = _find_eval_run(outputs, provider.model_name, seed, f"cp_e{epoch_label}")
    result = _read_eval_summary(run_dir)
    if result is None:
        raise RuntimeError(f"no rollout_summary.json under {run_dir}")
    return result


def _telemetry_at_epoch(curves_path: Path, epoch: int) -> dict:
    for line in curves_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("epoch") == epoch:
            return {
                "val_loss": row.get("val/loss_action"),
                "nmi": row.get("val/phase_expert_nmi"),
                "entropy": row.get("val/routing_entropy"),
                "top1_balance": row.get("val/top1_balance_score"),
                "top1_collapse": row.get("val/top1_collapse_rate"),
            }
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--ckpt-cadence", type=int, default=10,
                        help="Save a checkpoint every N stage-2 epochs (default 10).")
    parser.add_argument("--eval-epochs", default="10,30,100,200,best")
    parser.add_argument("--outputs", default="outputs/surgical")
    parser.add_argument("--skip-provider", action="store_true",
                        help="Reuse the existing stage-1 run only; fail if absent.")
    parser.add_argument("--force-evals", action="store_true",
                        help="Re-run evals whose run already exists.")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    eval_epochs = [e for e in args.eval_epochs.split(",") if e]
    outputs = (PROJECT_ROOT / args.outputs).resolve()
    logs = outputs / "_sweep" / "logs"
    defaults = load_protocol(PROJECT_ROOT / "experiments" / "lift_ablation.json").defaults
    provider = load_protocol(PROJECT_ROOT / "experiments" / "lift_ablation.json").method_by_name("phaseforge")
    if provider is None:
        print("[FAIL] lift_ablation manifest has no 'phaseforge' provider cell")
        return 1

    print(f"[sweep] seeds={seeds} stage1={args.stage1_epochs} stage2={args.epochs} "
          f"eval_epochs={eval_epochs} outputs={outputs}")

    evals: dict[str, dict] = {}
    corr: dict[str, float | None] = {}
    selected: dict[str, int | None] = {}
    start = time.time()

    for seed in seeds:
        prov_dir: Path | None = None
        try:
            prov_dir = _find_run_by_meta(_run_dir_base(outputs, provider, 1, seed), provider.name, seed, None)
        except RuntimeError:
            prov_dir = None
        if prov_dir is None:
            if args.skip_provider:
                print(f"[sweep] seed {seed}: --skip-provider but no stage-1 run found; aborting")
                return 1
            print(f"[sweep] seed {seed}: training phaseforge stage 1 ({args.stage1_epochs} epochs)")
            if args.dry_run:
                continue
            prov_dir = _train_stage1(
                provider, seed, outputs, defaults, args.stage1_epochs, logs, args.timeout
            )
        provider_ckpt = prov_dir / "checkpoints" / "checkpoint_best.pt"

        stage2_dir = _run_dir_base(outputs, provider, 2, seed)
        run_dir: Path | None = None
        if stage2_dir.is_dir():
            try:
                run_dir = _find_run_by_meta(stage2_dir, provider.name, seed, None)
            except RuntimeError:
                run_dir = None
        max_reg = max((int(e) for e in eval_epochs if e != "best"), default=0)
        if run_dir is not None and (run_dir / f"checkpoints/checkpoint_epoch_{max_reg:04d}.pt").is_file():
            print(f"[sweep] seed {seed}: stage 2 already complete ({run_dir.name})")
        else:
            print(f"[sweep] seed {seed}: training phaseforge stage 2 ({args.epochs} epochs, "
                  f"every {args.ckpt_cadence}-epoch checkpoints)")
            if args.dry_run:
                continue
            run_dir = _train_stage2(
                provider, seed, outputs, defaults, args.epochs, args.ckpt_cadence,
                provider_ckpt, logs, args.timeout,
            )
            run_dir = _find_run_by_meta(stage2_dir, provider.name, seed, None)

        curves = run_dir / "metrics" / "training_curves.jsonl"
        summary_path = run_dir / "metrics" / "summary.json"
        best_epoch = json.loads(summary_path.read_text(encoding="utf-8")).get("best_epoch")
        selected[str(seed)] = best_epoch

        evals[str(seed)] = {}
        for epoch_label in eval_epochs:
            if epoch_label == "best":
                ckpt = run_dir / "checkpoints" / "checkpoint_best.pt"
                telemetry = _telemetry_at_epoch(curves, best_epoch) if best_epoch is not None else {}
            else:
                ckpt = run_dir / "checkpoints" / f"checkpoint_epoch_{int(epoch_label):04d}.pt"
                telemetry = _telemetry_at_epoch(curves, int(epoch_label))
            if not ckpt.is_file():
                print(f"[sweep] seed {seed}: skipping epoch {epoch_label} (no checkpoint)")
                continue
            if args.dry_run:
                continue
            existing = None
            if not args.force_evals:
                try:
                    existing = _find_eval_run(outputs, provider.model_name, seed, f"cp_e{epoch_label}")
                except RuntimeError:
                    existing = None
            result = None
            if existing is not None:
                result = _read_eval_summary(existing)
                if result is not None:
                    print(f"[sweep] seed {seed}: reusing eval epoch {epoch_label} ({existing.name})")
            if result is None:
                print(f"[sweep] seed {seed}: evaluating epoch {epoch_label} "
                      f"({time.time() - start:.0f}s elapsed)")
                result = _eval_epoch(
                    outputs, provider, seed, ckpt, epoch_label, defaults, logs, args.timeout
                )
            result["val_loss"] = telemetry.get("val_loss")
            result["nmi"] = telemetry.get("nmi")
            result["entropy"] = telemetry.get("entropy")
            result["top1_balance"] = telemetry.get("top1_balance")
            result["top1_collapse"] = telemetry.get("top1_collapse")
            evals[str(seed)][epoch_label] = result

    if args.dry_run:
        print("[sweep] dry run — no commands executed")
        return 0

    if not any(evals.get(str(s)) for s in seeds):
        print("[sweep] FAIL: no evals produced for any seed (checkpoints missing?)")
        return 1

    for seed in seeds:
        rows = evals.get(str(seed), {})
        xs = [r["val_loss"] for r in rows.values() if r.get("val_loss") is not None and r.get("sr") is not None]
        ys = [r["sr"] for r in rows.values() if r.get("val_loss") is not None and r.get("sr") is not None]
        corr[str(seed)] = _pearson(xs, ys)

    pooled_x: list[float] = []
    pooled_y: list[float] = []
    for seed_rows in evals.values():
        for r in seed_rows.values():
            if r.get("val_loss") is not None and r.get("sr") is not None:
                pooled_x.append(r["val_loss"])
                pooled_y.append(r["sr"])
    corr["pooled"] = _pearson(pooled_x, pooled_y)

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": seeds,
        "ckpt_cadence": args.ckpt_cadence,
        "eval_epochs": eval_epochs,
        "evals": evals,
        "corr_val_sr": corr,
        "selected_epoch": selected,
    }
    findings_path = PROJECT_ROOT / FINDINGS_DIR / "checkpoint_sweep.json"
    findings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _render_report(payload)

    print(f"[sweep] done in {time.time() - start:.0f}s -> {findings_path}")
    return 0


def _render_report(payload: dict) -> None:
    lines = [
        "# Wave A1 — Checkpoint sweep: SR vs epoch (paired 50-episode bank)",
        "",
        f"Created {payload['created']} on branch `surgical-cpu-analysis`.",
        "",
        f"Config: seeds={payload['seeds']}, stage-2 checkpoint cadence={payload['ckpt_cadence']}, "
        f"evaluated epochs={payload['eval_epochs']} (best = protocol-selected val-loss checkpoint).",
        "",
    ]
    for seed in payload["seeds"]:
        rows = payload["evals"].get(str(seed), {})
        lines += [f"## Seed {seed}", "", "| epoch | SR | CI95 | val loss | NMI | entropy | top1 bal | collapse |", "|---|---|---|---|---|---|---|---|"]
        for label, r in rows.items():
            def fmt(x, d=3):
                return "-" if x is None else f"{x:.{d}f}"
            lines.append(
                f"| {label} | {fmt(r.get('sr'))} | {fmt(r.get('ci_low'))}-{fmt(r.get('ci_high'))} "
                f"| {fmt(r.get('val_loss'), 4)} | {fmt(r.get('nmi'))} | {fmt(r.get('entropy'))} "
                f"| {fmt(r.get('top1_balance'))} | {fmt(r.get('top1_collapse'))} |"
            )
        sel = payload["selected_epoch"].get(str(seed))
        sr_at_sel = rows.get("best", {}).get("sr")
        peak = max((r["sr"] for r in rows.values() if r.get("sr") is not None), default=None)
        peak_label = next((l for l, r in rows.items() if r.get("sr") == peak), "-")
        lines += [
            "",
            f"- Protocol-selected checkpoint (best val loss): epoch **{sel}** → SR **{sr_at_sel}**",
            f"- Peak SR across evaluated epochs: **{peak}** at epoch **{peak_label}**",
            f"- corr(val action MSE, SR): **{payload['corr_val_sr'].get(str(seed))}**",
            "",
        ]
    lines += [
        "## Pooled",
        "",
        f"- corr(val action MSE, SR) across all seeds/epochs: **{payload['corr_val_sr'].get('pooled')}**",
        "",
        "## Interpretation checklist",
        "",
        "- If SR is sharply peaked around one epoch → checkpoint selection is a major bottleneck.",
        "- If SR is stable across a broad range → checkpoint noise is not the main problem.",
        "- If lowest val MSE and highest SR occur at different epochs → change model selection, not architecture.",
        "",
    ]
    report = PROJECT_ROOT / REPORT_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())