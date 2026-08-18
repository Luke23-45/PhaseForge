"""Wave B3+B4: balance coefficient and router noise grid.

Trains stage-2 only with one hyperparameter changed per cell; rollout-evaluates
the best-epoch checkpoint on the frozen reset bank.

Outputs:
    outputs/surgical/_findings/ablation_grid.json
    docs/dev/findings/ablation_grid.md
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
REPORT_PATH = Path("docs/dev/findings/ablation_grid.md")

GRID = [
    ("bc",  "balance_coeff",  [0.0, 0.01, 0.1]),
    ("rn",  "noise_std",      [0.0, 0.1, 0.5]),
]


def _step(method, seed: int, stage: int | None) -> Step:
    return Step(kind="train" if stage is not None else "eval", method=method, seed=seed, stage=stage)


def _train_stage2(provider, seed: int, outputs: Path, defaults, epochs: int, ckpt_path: Path, logs: Path, timeout: int) -> Path:
    quiet = list(TRAIN_QUIET) + list(STAGE2_QUIET) + [
        f"train.epochs={epochs}",
        "train.checkpoint.every_n_epochs=10",
    ]
    step = _step(provider, seed, 2)
    cmd = train_command(step, outputs_base=outputs, defaults=defaults + tuple(quiet), ckpt_path=ckpt_path)
    _execute(["phaseforge-train", "project.log_level=WARNING"] + cmd[1:],
             logs / f"stage2_seed{seed}.log", timeout)
    return _find_run_by_meta(_run_dir_base(outputs, provider, 2, seed), provider.name, seed, None)


def _eval_best(outputs: Path, provider, seed: int, ckpt_path: Path, tag: str, defaults, logs: Path, timeout: int) -> dict:
    step = _step(provider, seed, None)
    cmd = eval_command(step, ckpt_path=ckpt_path, outputs_base=outputs, defaults=defaults + EVAL_QUIET)
    _execute(["phaseforge-eval", "project.log_level=WARNING"] + cmd[1:],
             logs / f"eval_seed{seed}.log", timeout)
    run_dir = _find_run_by_meta(outputs / "eval" / provider.model_name / f"seed{seed}", provider.name, seed, tag)
    summary = run_dir / "rollout_summary.json"
    data = json.loads(summary.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    return {
        "sr": metrics.get("eval/rollout/success_rate"),
        "ci_low": metrics.get("eval/rollout/wilson_ci95_low"),
        "ci_high": metrics.get("eval/rollout/wilson_ci95_high"),
    }


def _find_provider_ckpt(outputs: Path, provider, seed: int) -> Path:
    prov_dir = _find_run_by_meta(_run_dir_base(outputs, provider, 1, seed), provider.name, seed, None)
    return prov_dir / "checkpoints" / "checkpoint_best.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--outputs", default="outputs/surgical")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    outputs = (PROJECT_ROOT / args.outputs).resolve()
    logs = outputs / "_ablation_grid" / "logs"
    protocol = load_protocol(PROJECT_ROOT / "experiments" / "lift_ablation.json")
    defaults = protocol.defaults
    base_provider = protocol.method_by_name("phaseforge")
    if base_provider is None:
        print("[grid] lift_ablation manifest has no 'phaseforge' provider cell")
        return 1

    print(f"[grid] seeds={seeds} epochs={args.epochs} outputs={outputs}")
    results: dict[str, dict[str, dict[str, dict]]] = {ax: {} for ax, _, _ in GRID}
    start = time.time()

    for seed in seeds:
        try:
            provider_ckpt = _find_provider_ckpt(outputs, base_provider, seed)
        except RuntimeError as exc:
            print(f"[grid] seed {seed}: {exc} — run Wave A1 (checkpoint_sweep) first")
            continue
        for axis, hp, values in GRID:
            for value in values:
                short = f"{axis}{str(value).replace('.', '')}"
                tag = short
                provider = dataclasses.replace(
                    base_provider,
                    name=f"phaseforge_{axis}_{short}",
                    tag=tag,
                    overrides=[f"models.router.{hp}={value}"],
                )
                try:
                    run_dir = _find_run_by_meta(_run_dir_base(outputs, provider, 2, seed), provider.name, seed, tag)
                    print(f"[grid] seed {seed} {axis}={value}: reusing {run_dir.name}")
                except RuntimeError:
                    print(f"[grid] seed {seed} {axis}={value}: training stage 2")
                    if args.dry_run:
                        continue
                    _train_stage2(provider, seed, outputs, defaults, args.epochs, provider_ckpt, logs, args.timeout)
                    run_dir = _find_run_by_meta(_run_dir_base(outputs, provider, 2, seed), provider.name, seed, tag)
                best_ckpt = run_dir / "checkpoints" / "checkpoint_best.pt"
                if not best_ckpt.is_file():
                    print(f"[grid] seed {seed} {axis}={value}: no checkpoint_best.pt, skipping eval")
                    continue
                if args.dry_run:
                    continue
                results[axis].setdefault(str(value), {})[str(seed)] = _eval_best(
                    outputs, provider, seed, best_ckpt, tag, defaults, logs, args.timeout
                )

    if args.dry_run:
        print("[grid] dry run — no commands executed")
        return 0

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": seeds,
        "grid": [{"axis": a, "hp": h, "values": v} for a, h, v in GRID],
        "results": results,
    }
    findings_path = PROJECT_ROOT / FINDINGS_DIR / "ablation_grid.json"
    findings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _render_report(payload)
    print(f"[grid] done in {time.time() - start:.0f}s -> {findings_path}")
    return 0


def _render_report(payload: dict) -> None:
    lines = [
        "# Wave B3+B4 — Balance coefficient and router noise ablation",
        "",
        f"Created {payload['created']} on branch `surgical-cpu-analysis`.",
        "",
    ]
    for axis_entry in payload["grid"]:
        axis = axis_entry["axis"]
        hp = axis_entry["hp"]
        lines += [f"## {axis} ({hp})", "", f"| value | mean SR | per-seed SR |", "|---|---|---|"]
        for value in axis_entry["values"]:
            key = str(value)
            per_seed = payload["results"].get(axis, {}).get(key, {})
            srs = [r["sr"] for r in per_seed.values() if r.get("sr") is not None]
            mean_sr = sum(srs) / len(srs) if srs else None
            per_seed_str = ", ".join(
                f"seed{s}={r['sr']:.3f}" if r.get("sr") is not None else f"seed{s}=null"
                for s, r in sorted(per_seed.items(), key=lambda kv: int(kv[0]))
            )
            lines.append(
                f"| {value} | {mean_sr if mean_sr is None else f'{mean_sr:.3f}'} | {per_seed_str} |"
            )
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "- If SR is flat across the grid, neither knob matters on the locked protocol.",
        "- The defaults (`balance_coeff=0.01`, `noise_std=0.1`) appear as the middle value",
        "  in each axis.",
        "",
    ]
    report = PROJECT_ROOT / REPORT_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
