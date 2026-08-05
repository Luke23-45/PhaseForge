"""Run 3-seed rollout evaluation for all PhaseForge model variants.

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

# Model config names and their checkpoint paths.
# If the checkpoint path is None, the latest best checkpoint is auto-detected
# from outputs/<model_name>/stage<N>/ (see STAGES below).
MODELS: list[tuple[str, str | None]] = [
    ("bc", None),
    ("phaseforge", None),
    ("scratch_moe", None),
    ("warmstart_moe", None),
    ("oracle_moe", None),
]

# Stage whose best checkpoint should be evaluated for each model.
# BC trains in Stage 1 only; all MoE variants are evaluated after Stage 2.
STAGES: dict[str, int] = {
    "bc": 1,
    "phaseforge": 2,
    "scratch_moe": 2,
    "warmstart_moe": 2,
    "oracle_moe": 2,
}

SEEDS = [42, 43, 44]

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def find_latest_checkpoint(model_name: str, stage: int = 2) -> Path | None:
    """Find the most recent checkpoint_best.pt for a model+stage.

    ``resolve_alias=False`` on purpose: aliases (e.g. warmstart_moe -> bc)
    only apply to Stage 1 pretraining checkpoints. For evaluation we want
    the model's OWN Stage 2 weights under outputs/warmstart_moe/stage2/.
    """
    base = Path("outputs") / model_name / f"stage{stage}"
    if not base.is_dir():
        return None
    runs = sorted(base.iterdir(), reverse=True)
    for run in runs:
        if not run.is_dir():
            continue
        ckpt = run / "checkpoints" / "checkpoint_best.pt"
        if ckpt.is_file():
            return ckpt
    return None


def run_eval(model_name: str, ckpt_path: Path | None, seed: int) -> dict:
    """Run a single evaluation and return the parsed JSON results."""
    cmd = [
        "phaseforge-eval",
        f"models={model_name}",
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
    eval_base = Path("outputs/eval") / model_name
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

    for model_name, ckpt_path_str in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        stage = STAGES.get(model_name, 2)
        ckpt_path = (
            Path(ckpt_path_str)
            if ckpt_path_str
            else find_latest_checkpoint(model_name, stage=stage)
        )

        if ckpt_path is None or not ckpt_path.exists():
            print(f"  No checkpoint found for {model_name}, skipping.")
            continue

        print(f"  Checkpoint: {ckpt_path}")

        per_seed_results: list[dict] = []

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            result = run_eval(model_name, ckpt_path, seed)
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

    # Print final table
    header = (
        f"\n{'Model':<20} {'Average':<15} {'Spatial':<15} "
        f"{'Object':<15} {'Goal':<15} {'Long':<15}"
    )
    print(header)
    print("-" * 90)
    for model_name, summary in all_results.items():
        avg = f"{summary['mean_success_rate']:.4f} ± {summary['std_success_rate']:.4f}"
        spatial = (
            f"{summary.get('libero_spatial_mean', 0):.4f} "
            f"± {summary.get('libero_spatial_std', 0):.4f}"
        )
        obj = (
            f"{summary.get('libero_object_mean', 0):.4f} "
            f"± {summary.get('libero_object_std', 0):.4f}"
        )
        goal = (
            f"{summary.get('libero_goal_mean', 0):.4f} "
            f"± {summary.get('libero_goal_std', 0):.4f}"
        )
        long_ = (
            f"{summary.get('libero_10_mean', 0):.4f} "
            f"± {summary.get('libero_10_std', 0):.4f}"
        )
        print(f"{model_name:<20} {avg:<15} {spatial:<15} {obj:<15} {goal:<15} {long_:<15}")


if __name__ == "__main__":
    main()
