"""Diagnose the phase head: classification error bucketed by distance to the
nearest phase transition, evaluated offline on the validation split.

Usage:
  uv run python scripts/phase_boundary_diagnostic.py <stage1-run-dir> [--checkpoint PATH]

Reads the run's resolved_config.yaml, enables the phase_boundary_error metric
(diagnostic, off by default), rebuilds the model from the run checkpoint and
prints the per-bucket error rates. Writes the breakdown to
``phase_boundary_diagnostic.json`` next to the run.

Diagnostic intent (stage-1 seed-lottery): errors clustering at distance 0
indicate boundary label noise; errors spread uniformly across distances
indicate auxiliary-head overfitting. CPU-only (no robosuite needed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from phaseforge.cli import build_eval_model
from phaseforge.evaluations.runners.offline_evaluator import OfflineEvaluator
from phaseforge.models.base import BaseManipulationModel
from phaseforge.utils.registry import build_data_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dir", type=Path, help="stage-1 run directory (with resolved_config.yaml)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint to evaluate (default: run_dir/checkpoints/checkpoint_best.pt)",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.is_file():
        sys.exit(f"resolved_config.yaml not found under {run_dir}")
    ckpt = args.checkpoint or run_dir / "checkpoints" / "checkpoint_best.pt"
    if not ckpt.is_file():
        sys.exit(f"checkpoint not found: {ckpt}")

    cfg = cast(DictConfig, OmegaConf.load(cfg_path))
    cfg.project.device = "cpu"
    cfg.train.stage1_ckpt_path = str(ckpt)
    if cfg.eval.get("task") is None:
        cfg.eval.task = {}
    if cfg.eval.task.get("phase_boundary_error") is None:
        cfg.eval.task.phase_boundary_error = {"enabled": True, "bucket_edges": [1, 3, 6, 11]}
    cfg.eval.task.phase_boundary_error.enabled = True
    cfg.eval.mode = "offline"

    pipeline = build_data_pipeline(cfg)
    loaders = pipeline.run()
    val_loader = loaders.get("val") or loaders.get("test")
    if val_loader is None:
        sys.exit("no validation/test split available for this data config")

    model = cast(BaseManipulationModel, build_eval_model(cfg))
    model.to(torch.device("cpu"))
    model.eval()

    evaluator = OfflineEvaluator(cfg=cfg, model=model, dataloader=val_loader)
    results = evaluator.run()

    phase_err = {k: v for k, v in results.items() if k.startswith("eval/phase_err_")}
    if not phase_err:
        print("phase_boundary_error produced no keys (no phase logits / no trajectory ids).")
    for key in sorted(phase_err):
        print(f"  {key}: {phase_err[key]}")

    out_path = run_dir / "phase_boundary_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(phase_err, f, indent=2, sort_keys=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
