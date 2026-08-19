"""Wave A3: validation-set sampling variance.

For each seed, repeat the stage-2 training four times with a different
validation split seed (``data.split.seed`` with ``data.split.use_dataset_filters=false``,
because the HDF5 train/valid filters would otherwise short-circuit the split
RNG and fix the fold); rollout-evaluate the best-epoch checkpoint on the
frozen reset bank (50 episodes). The spread of the four SRs quantifies how
much of the seed-to-seed variance is ``checkpoint selection noise from a
20-trajectory val split`` versus ``inherent stage-2 variance``.

Outputs:
    outputs/surgical/_findings/validation_bank.json
    docs/dev/findings/validation_bank.md
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
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from smoke_matrix import (  # noqa: E402
    EVAL_QUIET,
    STAGE2_QUIET,
    TRAIN_QUIET,
    _execute,
    _find_provider_ckpt,
    _find_run_by_meta,
    _run_dir_base,
)

FINDINGS_DIR = Path("outputs/surgical/_findings")
REPORT_PATH = Path("docs/dev/findings/validation_bank.md")

BANK_SEEDS = [142, 242, 342, 442]


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
    best_epoch = None
    train_run = _find_run_by_meta(
        _run_dir_base(outputs, provider, 2, seed), provider.name, seed, None
    )
    summary_path = train_run / "metrics" / "summary.json"
    if summary_path.is_file():
        best_epoch = json.loads(summary_path.read_text(encoding="utf-8")).get("best_epoch")
    return {
        "sr": metrics.get("eval/rollout/success_rate"),
        "ci_low": metrics.get("eval/rollout/wilson_ci95_low"),
        "ci_high": metrics.get("eval/rollout/wilson_ci95_high"),
        "best_epoch": best_epoch,
    }


def _run_has_filters_disabled(run_dir: Path) -> bool:
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.is_file():
        return False
    try:
        cfg = OmegaConf.load(str(cfg_path))
        return not bool(cfg.get("data", {}).get("split", {}).get("use_dataset_filters", True))
    except Exception:
        return False


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
    logs = outputs / "_validation_bank" / "logs"
    protocol = load_protocol(PROJECT_ROOT / "experiments" / "lift_ablation.json")
    defaults = protocol.defaults
    base_provider = protocol.method_by_name("phaseforge")
    if base_provider is None:
        print("[vbank] lift_ablation manifest has no 'phaseforge' provider cell")
        return 1

    print(f"[vbank] seeds={seeds} banks={BANK_SEEDS} epochs={args.epochs} outputs={outputs}")
    results: dict[str, dict[str, dict]] = {str(b): {} for b in BANK_SEEDS}
    start = time.time()

    for seed in seeds:
        try:
            provider_ckpt = _find_provider_ckpt(outputs, base_provider, seed)
        except RuntimeError as exc:
            print(f"[vbank] seed {seed}: {exc} — run Wave A1 (checkpoint_sweep) first")
            continue
        for bank in BANK_SEEDS:
            tag = f"vbank_b{bank}"
            provider = dataclasses.replace(
                base_provider,
                name=f"phaseforge_vbank_{bank}",
                tag=tag,
                overrides=[
                    f"data.split.seed={bank}",
                    "data.split.use_dataset_filters=false",
                ],
            )
            try:
                run_dir = _find_run_by_meta(_run_dir_base(outputs, provider, 2, seed), provider.name, seed, tag)
                if not _run_has_filters_disabled(run_dir):
                    print(f"[vbank] seed {seed} bank {bank}: stale run {run_dir.name} (dataset filters enabled), retraining")
                    raise RuntimeError("stale run without use_dataset_filters=false")
                print(f"[vbank] seed {seed} bank {bank}: reusing {run_dir.name}")
            except RuntimeError:
                print(f"[vbank] seed {seed} bank {bank}: training stage 2")
                if args.dry_run:
                    continue
                _train_stage2(provider, seed, outputs, defaults, args.epochs, provider_ckpt, logs, args.timeout)
                run_dir = _find_run_by_meta(_run_dir_base(outputs, provider, 2, seed), provider.name, seed, tag)
            best_ckpt = run_dir / "checkpoints" / "checkpoint_best.pt"
            if not best_ckpt.is_file():
                print(f"[vbank] seed {seed} bank {bank}: no checkpoint_best.pt, skipping eval")
                continue
            if args.dry_run:
                continue
            results[str(bank)][str(seed)] = _eval_best(
                outputs, provider, seed, best_ckpt, tag, defaults, logs, args.timeout
            )

    if args.dry_run:
        print("[vbank] dry run — no commands executed")
        return 0

    if not any(results[b] for b in BANK_SEEDS):
        print(f"[vbank] FAIL: no (bank, seed) cell produced results (provider stage-1 missing?)")
        return 1

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": seeds,
        "bank_seeds": BANK_SEEDS,
        "results": results,
    }
    findings_path = PROJECT_ROOT / FINDINGS_DIR / "validation_bank.json"
    findings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _render_report(payload)
    print(f"[vbank] done in {time.time() - start:.0f}s -> {findings_path}")
    return 0


def _render_report(payload: dict) -> None:
    lines = [
        "# Wave A3 — Validation-bank sampling variance",
        "",
        f"Created {payload['created']} on branch `surgical-cpu-analysis`.",
        "",
        "Each cell is a separate stage-2 run with a different ``data.split.seed``.",
        "Best-epoch checkpoint is the val-loss monitor on that split.",
        "",
        "| seed | bank | best_epoch | SR | CI95 |",
        "|---|---|---|---|---|",
    ]
    for bank in payload["bank_seeds"]:
        per_seed = payload["results"].get(str(bank), {})
        for s in payload["seeds"]:
            r = per_seed.get(str(s), {})
            sr = r.get("sr")
            ci = "-"
            if r.get("ci_low") is not None and r.get("ci_high") is not None:
                ci = f"{r['ci_low']:.3f}-{r['ci_high']:.3f}"
            sr_str = f"{sr:.3f}" if sr is not None else "-"
            lines.append(
                f"| {s} | {bank} | {r.get('best_epoch') if r.get('best_epoch') is not None else '-'} | "
                f"{sr_str} | {ci} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Spread of SR across banks (per seed) = checkpoint-selection noise floor.",
        "- Compare against the SR spread across epoch checkpoints (Wave A1).",
        "",
    ]
    report = PROJECT_ROOT / REPORT_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
