"""Run 3-seed rollout evaluation for all PhaseForge model variants.

Each seed's evaluation uses the checkpoint TRAINED with that same seed
(seed-matched lookup via ``phaseforge.utils.config.find_latest_checkpoint``).

The suites are the Decision-2 set only: ``libero_90`` (in-distribution, ID)
and ``libero_10`` (labeled zero-shot row). The per-suite mean/std in
``outputs/eval/final_results.json`` reports ID vs OOD separately; every
rollout run's ``eval_results.json`` additionally declares the ID/OOD role
of each suite (B4).

Usage:
    uv sync --extra rollout          # one-time: installs libero + robosuite
    uv run python scripts/run_multi_seed_eval.py

Requires:
    - All model checkpoints to exist at the paths below (update as needed)
    - libero + robosuite installed (pip install -e ".[rollout]")
    - Data cache built (run training first)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from phaseforge.utils.config import find_latest_checkpoint

# (model name, Hydra config path, explicit checkpoint path or None).
# The config path is REQUIRED: model configs live under config/models/…
# (e.g. ``baselines/bc``); a bare name like ``bc`` does not resolve. If the
# checkpoint path is None, the seed-matched best checkpoint is auto-detected
# from outputs/<model_name>/stage<N>/ (see STAGES below).
MODELS: list[tuple[str, str, str | None]] = [
    ("bc", "baselines/bc", None),
    ("phaseforge", "phaseforge", None),
    ("scratch_moe", "baselines/scratch_moe", None),
    ("warmstart_moe", "baselines/warmstart_moe", None),
    ("oracle_moe", "baselines/oracle_moe", None),
    ("phase_pretrain_random_router", "baselines/phase_pretrain_random_router", None),
    ("plain_encoder_phase_bootstrap", "baselines/plain_encoder_phase_bootstrap", None),
    ("teacher_forced", "baselines/teacher_forced", None),
]

# Stage whose best checkpoint should be evaluated for each model.
# BC trains in Stage 1 only; all MoE variants are evaluated after Stage 2.
STAGES: dict[str, int] = {
    "bc": 1,
    "phaseforge": 2,
    "scratch_moe": 2,
    "warmstart_moe": 2,
    "oracle_moe": 2,
    "phase_pretrain_random_router": 2,
    "plain_encoder_phase_bootstrap": 2,
    "teacher_forced": 2,
}

SEEDS = [42, 43, 44]

# Decision 2 (issues register A2): only the in-distribution suite and the
# labeled zero-shot suite are evaluated.
SUITES = ["libero_90", "libero_10"]


def run_eval(model_cfg: str, ckpt_path: Path | None, seed: int) -> dict:
    """Run a single evaluation and return the parsed JSON results."""
    cmd = [
        "phaseforge-eval",
        f"models={model_cfg}",
        "eval=rollout",
        f"project.seed={seed}",
    ]
    if ckpt_path is not None:
        cmd.append(f"train.stage1_ckpt_path={ckpt_path}")

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}): {result.stderr[:500]}")
        return {"error": True, "stderr": result.stderr[:500]}

    # Parse results from the output JSON file
    # The eval command writes to outputs/eval/{model_name}/{run_id}/eval_results.json
    # Find the latest one
    eval_base = Path("outputs/eval") / model_cfg.split("/")[-1]
    if not eval_base.is_dir():
        print(f"  WARNING: no output directory found at {eval_base}")
        return {"error": True}

    runs = sorted(eval_base.iterdir(), reverse=True)
    if not runs:
        return {"error": True}

    results_path = runs[0] / "eval_results.json"
    if not results_path.is_file():
        print(f"  WARNING: {results_path} not found")
        return {"error": True}

    with open(results_path) as f:
        return json.load(f)


def main() -> None:
    all_results: dict[str, dict] = {}

    for model_name, model_cfg, ckpt_path_str in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        stage = STAGES.get(model_name, 2)
        per_seed_results: list[dict] = []

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            ckpt_path = (
                Path(ckpt_path_str)
                if ckpt_path_str
                else find_latest_checkpoint(model_name, stage=stage, resolve_alias=False, seed=seed)
            )

            if ckpt_path is None or not ckpt_path.exists():
                print(f"  No checkpoint found for {model_name} (seed {seed}), skipping.")
                continue

            print(f"    Checkpoint: {ckpt_path}")
            result = run_eval(model_cfg, ckpt_path, seed)
            per_seed_results.append(result)

        # Aggregate success rates across seeds
        suite_rates: dict[str, list[float]] = {s: [] for s in SUITES}
        overall_rates: list[float] = []

        for result in per_seed_results:
            if result.get("error"):
                continue
            overall_rates.append(float(result.get("eval/success_rate", 0.0)))
            for suite in SUITES:
                key = f"eval/success_rate/{suite}"
                if key in result:
                    suite_rates[suite].append(float(result[key]))

        if overall_rates:
            summary: dict[str, float | dict] = {
                "mean_success_rate": float(np.mean(overall_rates)),
                "std_success_rate": float(np.std(overall_rates)),
                "per_seed_success_rates": overall_rates,
                "suite_roles": {
                    "libero_90": "in-distribution",
                    "libero_10": "zero-shot (labeled)",
                },
            }
            for suite in SUITES:
                if suite_rates[suite]:
                    summary[f"{suite}_mean"] = float(np.mean(suite_rates[suite]))
                    summary[f"{suite}_std"] = float(np.std(suite_rates[suite]))
            all_results[model_name] = summary

            print(
                f"\n  Results: {summary['mean_success_rate']:.4f} "
                f"± {summary['std_success_rate']:.4f}"
            )
            for suite in SUITES:
                if suite in suite_rates and suite_rates[suite]:
                    m = np.mean(suite_rates[suite])
                    s = np.std(suite_rates[suite])
                    print(f"    {suite}: {m:.4f} ± {s:.4f}")

    # Save final aggregated results
    output_path = Path("outputs/eval/final_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Final results saved to {output_path}")
    print(f"{'='*60}")

    # Print final table (ID = libero_90, OOD = libero_10 per Decision 2)
    header = f"\n{'Model':<32} {'Overall':<15}" + "".join(
        f" {suite:<15}" for suite in SUITES
    )
    print(header)
    print("-" * (32 + 15 + 15 * len(SUITES)))
    for model_name, summary in all_results.items():
        row = (
            f"{model_name:<32} "
            f"{summary['mean_success_rate']:.4f} ± {summary['std_success_rate']:.4f}"
        )
        for suite in SUITES:
            m = summary.get(f"{suite}_mean")
            s = summary.get(f"{suite}_std")
            cell = f"{m:.4f} ± {s:.4f}" if m is not None else "n/a"
            row += f" {cell:<15}"
        print(row)


if __name__ == "__main__":
    main()
