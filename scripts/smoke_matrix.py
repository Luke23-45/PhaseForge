"""Fast smoke run for the new evaluation cells before the final matrix.

Trains every new cell (Wave-1/2 lift_ablation additions + the five-task
bc_rnn rows) for a handful of epochs on the real data and runs the offline
metric evaluator, so composition, bootstrap, checkpoint and metric-path
errors surface in minutes instead of after the 100/200-epoch runs.

Usage:
    python scripts/smoke_matrix.py [--seed 42] [--outputs outputs_smoke]
        [--epochs1 1] [--epochs2 2] [--workers 4] [--timeout 900]
        [--only bc_large,pf_k3] [--dry-run]

Exit code 0 = every cell passed; 1 = at least one cell failed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from phaseforge.runner.commands import eval_command, train_command
from phaseforge.runner.protocol import Method, Step, load_protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NEW_ABLATION = {
    "bc_large",  # EXP-103
    "pf_spherical_kmeans",  # EXP-109
    "pf_kmeans",  # EXP-110
    "pf_phase_head",  # EXP-111
    "pf_random_random",  # EXP-112
    "pf_centroid_random",  # EXP-113
    "pf_spherical",  # EXP-114
    "pf_ft",  # EXP-115
    "pf_k3",  # EXP-201
    "pf_k12",  # EXP-202
    "pf_jitter_00",  # EXP-203
    "pf_jitter_10",  # EXP-204
    "pf_corrupt_25",  # EXP-205
    "pf_corrupt_50",  # EXP-206
    "pf_shuffle_control",  # EXP-207
    "warmstart_r50",  # EXP-210 (Wave 3 expert-init)
    "pf_one_warm_plus_random",  # EXP-211 (Wave 3 expert-init)
}

NEW_BC_RNN = {"bc_rnn"}

TRAIN_QUIET = (
    "data.num_workers=0",
    "data.persistent_workers=false",
    "data.pin_memory=false",
    "train.epoch_progressbar=false",
    "train.rich_progressbar=false",
    "train.log_every_n_steps=100000",
    "train.checkpoint.every_n_epochs=1",
)

STAGE2_QUIET = ("train.routing_log_every_n_steps=100000",)

EVAL_QUIET = (
    "data.num_workers=0",
    "data.persistent_workers=false",
    "data.pin_memory=false",
)


def _step(method: Method, seed: int, stage: int | None = None) -> Step:
    return Step(kind="train" if stage is not None else "eval", method=method, seed=seed, stage=stage)


def _run_dir_base(outputs: Path, method: Method, stage: int, seed: int) -> Path:
    return outputs / method.model_name / f"stage{stage}" / f"seed{seed}"


def _find_run_by_meta(base: Path, method_name: str, seed: int, tag: str | None) -> Path:
    """Find the run dir whose run_meta.json matches method+seed (+tag).

    Prefer the newest matching run; re-runs create new timestamped dirs.
    """
    if not base.is_dir():
        raise RuntimeError(f"no run dirs under {base}")
    matches: list[Path] = []
    for run in base.iterdir():
        if not run.is_dir():
            continue
        meta = run / "run_meta.json"
        if not meta.is_file():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("seed") != seed or data.get("method") != method_name:
            continue
        if tag is not None and data.get("tag") != tag:
            continue
        matches.append(run)
    if not matches:
        raise RuntimeError(
            f"no run_meta.json under {base} matching method={method_name} "
            f"seed={seed} tag={tag}"
        )
    return sorted(matches)[-1]


def _find_provider_ckpt(outputs: Path, provider, seed: int) -> Path:
    """Stage-1 best checkpoint, refusing incomplete runs (no checkpoint files)."""
    prov_dir = _find_run_by_meta(_run_dir_base(outputs, provider, 1, seed), provider.name, seed, None)
    ckpt = prov_dir / "checkpoints" / "checkpoint_best.pt"
    if not ckpt.is_file():
        raise RuntimeError(
            f"stage-1 run {prov_dir.name} is incomplete (missing checkpoints/checkpoint_best.pt)"
        )
    return ckpt


def _execute(argv: list[str], log_path: Path, timeout: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(
            argv, cwd=str(PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT,
            text=True, timeout=timeout,
        )
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
        raise RuntimeError(f"exit code {result.returncode}\n$ {' '.join(argv)}\n{tail}")


def train_cell(
    method: Method,
    seed: int,
    outputs: Path,
    defaults: tuple[str, ...],
    *,
    epochs: int,
    stage: int,
    ckpt_path: Path | None,
    logs: Path,
    timeout: int,
) -> Path:
    quiet = list(TRAIN_QUIET) + [f"train.epochs={epochs}"]
    if stage == 2:
        quiet += list(STAGE2_QUIET)
    step = _step(method, seed, stage)
    cmd = train_command(
        step, outputs_base=outputs, defaults=defaults + tuple(quiet), ckpt_path=ckpt_path
    )
    _execute(["phaseforge-train", "project.log_level=WARNING"] + cmd[1:],
             logs / f"{method.name}_stage{stage}.log", timeout)
    run_dir = _find_run_by_meta(_run_dir_base(outputs, method, stage, seed),
                                method.name, seed, method.output_tag)
    ckpt = run_dir / "checkpoints" / "checkpoint_best.pt"
    if not ckpt.is_file():
        raise RuntimeError(f"no checkpoint_best.pt under {run_dir}")
    return run_dir


def eval_cell(
    method: Method,
    seed: int,
    outputs: Path,
    defaults: tuple[str, ...],
    *,
    ckpt_path: Path,
    logs: Path,
    timeout: int,
) -> None:
    offline = dataclasses.replace(method, evaluate_mode="offline")
    step = _step(offline, seed)
    cmd = eval_command(step, ckpt_path=ckpt_path, outputs_base=outputs, defaults=defaults + EVAL_QUIET)
    _execute(["phaseforge-eval", "project.log_level=WARNING"] + cmd[1:],
             logs / f"{method.name}_eval.log", timeout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outputs", default="outputs_smoke")
    parser.add_argument("--epochs1", type=int, default=1)
    parser.add_argument("--epochs2", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--only", help="comma-separated subset of cell names")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    ablation = load_protocol(PROJECT_ROOT / "experiments" / "lift_ablation.json")
    five = load_protocol(PROJECT_ROOT / "experiments" / "five_task.json")
    outputs = (PROJECT_ROOT / args.outputs).resolve()
    logs = outputs / "_smoke" / "logs"
    seed = args.seed
    only = set(args.only.split(",")) if args.only else None

    def wanted(name: str) -> bool:
        return only is None or name in only

    targets: list[Method] = []
    for m in ablation.methods:
        if m.name in NEW_ABLATION and wanted(m.name):
            targets.append(m)
    for m in five.methods:
        if m.name in NEW_BC_RNN and wanted(m.name):
            targets.append(m)

    provider = ablation.method_by_name("phaseforge")
    if provider is None:
        print("[FAIL] lift_ablation manifest has no 'phaseforge' provider cell")
        return 1

    print(f"[smoke] {len(targets)} target cells: {[m.name for m in targets]}")
    if args.dry_run:
        return 0

    start = time.time()
    print("[smoke] provider: phaseforge stage1 (1 epoch)")
    try:
        prov_dir = train_cell(
            provider, seed, outputs, ablation.defaults,
            epochs=args.epochs1, stage=1, ckpt_path=None, logs=logs, timeout=args.timeout,
        )
    except Exception as exc:
        print(f"[FAIL] provider phaseforge stage1: {exc}")
        return 1
    provider_ckpt = prov_dir / "checkpoints" / "checkpoint_best.pt"
    print(f"[ok]   phaseforge stage1 -> {prov_dir.name}")

    def job(method: Method) -> tuple[str, str, float]:
        t0 = time.time()
        defaults = ablation.defaults if method.name != "bc_rnn" else five.defaults
        try:
            if method.stages == (1,):
                run_dir = train_cell(
                    method, seed, outputs, defaults,
                    epochs=args.epochs1, stage=1, ckpt_path=None, logs=logs, timeout=args.timeout,
                )
                ckpt = run_dir / "checkpoints" / "checkpoint_best.pt"
                eval_cell(method, seed, outputs, defaults, ckpt_path=ckpt, logs=logs, timeout=args.timeout)
                return method.name, "train+eval ok", time.time() - t0
            run_dir = train_cell(
                method, seed, outputs, ablation.defaults,
                epochs=args.epochs2, stage=2, ckpt_path=provider_ckpt, logs=logs, timeout=args.timeout,
            )
            ckpt = run_dir / "checkpoints" / "checkpoint_best.pt"
            eval_cell(method, seed, outputs, ablation.defaults, ckpt_path=ckpt, logs=logs, timeout=args.timeout)
            return method.name, "train+eval ok", time.time() - t0
        except Exception as exc:
            return method.name, f"FAILED: {exc}", time.time() - t0

    results: list[tuple[str, str, float]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, m) for m in targets]
        for fut in as_completed(futures):
            name, status, dur = fut.result()
            results.append((name, status, dur))
            tag = "[ok]  " if not status.startswith("FAILED") else "[FAIL]"
            print(f"{tag} {name:<22} {dur:6.1f}s  {status[:120]}")

    print("-" * 80)
    failed = [r for r in results if r[1].startswith("FAILED")]
    print(f"[smoke] {len(results) - len(failed)}/{len(results)} cells passed in {time.time() - start:.0f}s")
    for name, status, _ in failed:
        print(f"[FAIL] {name}: {status}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())