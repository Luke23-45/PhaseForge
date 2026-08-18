"""Master runner for the surgical-cpu-analysis investigation.

Executes the Wave A -> B -> C scripts in order on the configured seeds.
Each script writes its own ``outputs/surgical/_findings/<name>.json``
and the rendered markdown report under ``docs/dev/findings/``. Skips
scripts whose findings JSON already exists (set ``--force`` to
re-run).

Default cloud invocation:
    !uv run python scripts/experiments/run_all.py
        --waves A B C
        --seeds 42,43,44
        --outputs outputs/surgical

The runner launches each script via subprocess (``uv run python ...``)
so the CLI logs land on their own stderr streams.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts" / "experiments"
FINDINGS_DIR = PROJECT_ROOT / "outputs" / "surgical" / "_findings"


WAVE_SCRIPTS = {
    "A1": ("checkpoint_sweep.py",        "checkpoint_sweep.json"),
    "A2": ("sr_val_corr.py",             "sr_val_corr.json"),
    "A3": ("validation_bank.py",         "validation_bank.json"),
    "A4": ("specialization_matrix.py",   "specialization_matrix.json"),
    "A5": ("routing_counterfactuals.py", "routing_counterfactuals.json"),
    "B1": ("four_way_init.py",           "four_way_init.json"),
    "B2": ("expert_diversity.py",        "expert_diversity.json"),
    "B3_B4": ("ablation_grid.py",        "ablation_grid.json"),
    "C1": ("latent_geometry.py",         "latent_geometry.json"),
    "C2": ("failure_phase.py",           "failure_phase_<label>.json"),
}

FAILURE_PHASE_DIRS = ["outputs/part4/1", "outputs/part4/2", "outputs/part4/3", "outputs/part5"]


def _resolve_scripts(waves: list[str]) -> list[tuple[str, str, str]]:
    selected = []
    for wave in waves:
        for k, v in WAVE_SCRIPTS.items():
            if k.startswith(wave) or k == wave:
                selected.append((k, v[0], v[1]))
    seen = set()
    deduped = []
    for k, script, findings in selected:
        if k in seen:
            continue
        seen.add(k)
        deduped.append((k, script, findings))
    return deduped


def _run(label: str, script: str, extra: list[str], timeout: int, force: bool, uv: str) -> int:
    findings_path = FINDINGS_DIR / {
        "checkpoint_sweep.py": "checkpoint_sweep.json",
        "sr_val_corr.py": "sr_val_corr.json",
        "validation_bank.py": "validation_bank.json",
        "specialization_matrix.py": "specialization_matrix.json",
        "routing_counterfactuals.py": "routing_counterfactuals.json",
        "four_way_init.py": "four_way_init.json",
        "expert_diversity.py": "expert_diversity.json",
        "ablation_grid.py": "ablation_grid.json",
        "latent_geometry.py": "latent_geometry.json",
    }[script]
    if not force and findings_path.is_file():
        print(f"[run-all] {label}: skip ({findings_path.name} present)")
        return 0
    cmd = [uv, "run", "python", str(SCRIPTS / script), *extra]
    print(f"[run-all] {label}: {' '.join(cmd)}")
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=timeout)
    dt = time.time() - start
    print(f"[run-all] {label}: rc={proc.returncode} in {dt:.0f}s")
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waves", nargs="+", default=["A", "B", "C"],
                        help="Wave prefixes to run (default: all). Examples: A, A1, B3_B4.")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--outputs", default="outputs/surgical")
    parser.add_argument("--timeout", type=int, default=28800,
                        help="Per-script timeout in seconds (default 8h; the Wave A1 sweep is the long pole).")
    parser.add_argument("--force", action="store_true", help="Re-run scripts whose findings already exist.")
    parser.add_argument("--uv", default="uv", help="Path to uv (or python) executable.")
    parser.add_argument("--label", default="real_matrix", help="Label for the C2 failure_phase findings file.")
    args = parser.parse_args(argv)

    queue = _resolve_scripts(args.waves)
    if not queue:
        print("[run-all] no scripts selected")
        return 1

    common_extra = ["--seeds", args.seeds, "--outputs", args.outputs]
    overall_rc = 0
    for label, script, _findings in queue:
        if label == "C2":
            rc = _run_failure_phase(args, label, args.force, args.uv)
            if rc != 0:
                print(f"[run-all] FAIL at {label} (rc={rc}); continuing")
                overall_rc = rc
            continue
        rc = _run(label, script, common_extra, args.timeout, args.force, args.uv)
        if rc != 0:
            print(f"[run-all] FAIL at {label} (rc={rc}); continuing")
            overall_rc = rc
    return overall_rc


def _run_failure_phase(args, label: str, force: bool, uv: str) -> int:
    findings_path = FINDINGS_DIR / f"failure_phase_{args.label}.json"
    if not force and findings_path.is_file():
        print(f"[run-all] {label}: skip ({findings_path.name} present)")
        return 0
    cmd = [uv, "run", "python", str(SCRIPTS / "failure_phase.py"),
           "--eval-dirs", *FAILURE_PHASE_DIRS, "--label", args.label]
    print(f"[run-all] {label}: {' '.join(cmd)}")
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=args.timeout)
    print(f"[run-all] {label}: rc={proc.returncode} in {time.time() - start:.0f}s")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
