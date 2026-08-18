"""Oracle Routing Intervention Diagnostic.

Evaluates trained PhaseForge Stage-2 checkpoints under:
(a) Autonomous Learned Routing: standard inference via top-k router.
(b) Oracle Ground-Truth Routing: perfect phase-directed expert dispatch
    (expert e = phase z) on the exact same trained expert set.

Computes the true Routing Gap = MSE(learned) - MSE(oracle) to isolate
the cost of autonomous routing from expert capacity.

Writes outputs/phaseforge/stage2/oracle_diagnostic.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from phaseforge.utils.registry import build_data_pipeline, build_model

logger = logging.getLogger("phaseforge.oracle_diagnostic")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def evaluate_oracle_vs_routed(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device | str = "cuda",
    num_phases: int = 6,
) -> dict[str, Any]:
    """Compute per-phase and aggregate MSE for routed vs oracle GT dispatch."""
    model.eval()
    model.to(device)

    num_experts = len(model.moe_layer.experts)

    phase_routed_sq_err = torch.zeros(num_phases, dtype=torch.float64, device=device)
    phase_oracle_sq_err = torch.zeros(num_phases, dtype=torch.float64, device=device)
    phase_elem_counts = torch.zeros(num_phases, dtype=torch.float64, device=device)

    total_routed_sq_err = 0.0
    total_oracle_sq_err = 0.0
    total_elements = 0

    non_blocking = torch.device(device).type == "cuda"

    for batch in val_loader:
        state = batch["state"].to(device, non_blocking=non_blocking)
        target_action = batch["action"].to(device, non_blocking=non_blocking)
        phase = batch["phase"].to(device, non_blocking=non_blocking)

        if state.ndim == 3:
            state = state.view(-1, state.size(-1))
            target_action = target_action.view(-1, target_action.size(-1))
            phase = phase.view(-1)

        b, a_dim = target_action.shape
        total_elements += b * a_dim

        # 1. Learned routed forward. forward() takes a batch dict (reads
        #    batch["state"]) and returns ModelOutput.action_pred.
        model_out = model({"state": state})
        routed_pred = model_out.action_pred
        routed_sq_diff = (routed_pred - target_action) ** 2
        total_routed_sq_err += routed_sq_diff.sum().item()

        # 2. Oracle GT dispatch forward
        latent = model.encoder(state)
        oracle_pred = torch.zeros_like(target_action)
        for e_idx in range(num_experts):
            mask = (phase % num_experts) == e_idx
            if mask.any():
                oracle_pred[mask] = model.moe_layer.experts[e_idx](latent[mask])

        oracle_sq_diff = (oracle_pred - target_action) ** 2
        total_oracle_sq_err += oracle_sq_diff.sum().item()

        # Per-phase accumulators
        for p in range(num_phases):
            p_mask = phase == p
            if p_mask.any():
                n_p = p_mask.sum().item()
                phase_elem_counts[p] += n_p * a_dim
                phase_routed_sq_err[p] += routed_sq_diff[p_mask].sum().to(torch.float64)
                phase_oracle_sq_err[p] += oracle_sq_diff[p_mask].sum().to(torch.float64)

    # Calculate aggregate and per-phase metrics
    overall_routed_mse = total_routed_sq_err / max(total_elements, 1)
    overall_oracle_mse = total_oracle_sq_err / max(total_elements, 1)
    routing_gap = overall_routed_mse - overall_oracle_mse

    per_phase_metrics = {}
    for p in range(num_phases):
        cnt = phase_elem_counts[p].item()
        if cnt > 0:
            r_mse = float((phase_routed_sq_err[p] / cnt).item())
            o_mse = float((phase_oracle_sq_err[p] / cnt).item())
            per_phase_metrics[f"phase_{p}"] = {
                "routed_mse": r_mse,
                "oracle_mse": o_mse,
                "routing_gap": r_mse - o_mse,
                "samples": int(cnt // target_action.size(-1)),
            }
        else:
            per_phase_metrics[f"phase_{p}"] = {
                "routed_mse": float("nan"),
                "oracle_mse": float("nan"),
                "routing_gap": float("nan"),
                "samples": 0,
            }

    return {
        "overall_routed_mse": float(overall_routed_mse),
        "overall_oracle_mse": float(overall_oracle_mse),
        "overall_routing_gap": float(routing_gap),
        "relative_routing_overhead_pct": float(
            (routing_gap / max(overall_oracle_mse, 1e-8)) * 100.0
        ),
        "per_phase": per_phase_metrics,
    }


def run_oracle_diagnostics(
    output_dir: Path | str = "outputs",
    seeds: tuple[int, ...] = (42, 43, 44),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Run oracle routing diagnostic on PhaseForge checkpoints across all seeds."""
    out_base = Path(output_dir)
    config_dir = str(PROJECT_ROOT / "phaseforge" / "config")

    seed_results: dict[str, Any] = {}

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for seed in seeds:
            logger.info("Evaluating Oracle Diagnostic for seed %d...", seed)
            cfg: DictConfig = compose(
                config_name="main",
                overrides=[
                    "models=phaseforge",
                    "data=common",
                    "train=stage2",
                    f"project.seed={seed}",
                    f"project.output_dir={out_base}",
                ],
            )

            # Locate checkpoint
            ckpt_candidates = list(
                (out_base / "phaseforge" / "stage2" / f"seed{seed}").glob(
                    "**/checkpoints/checkpoint_best.pt"
                )
            )
            if not ckpt_candidates:
                # Try un-nested layout
                ckpt_candidates = list(
                    (out_base / "phaseforge" / "stage2").glob(
                        "**/checkpoints/checkpoint_best.pt"
                    )
                )

            if not ckpt_candidates:
                logger.warning(
                    "No PhaseForge stage2 checkpoint found for seed %d. Skipping.",
                    seed,
                )
                continue

            ckpt_path = sorted(ckpt_candidates)[-1]
            logger.info("Loading checkpoint: %s", ckpt_path)

            data_pipeline = build_data_pipeline(cfg)
            dataloaders = data_pipeline.get_dataloaders()
            val_loader = dataloaders["val"]

            model = build_model(cfg)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)

            diag = evaluate_oracle_vs_routed(model, val_loader, device=device)
            diag["checkpoint_path"] = str(ckpt_path)
            seed_results[f"seed_{seed}"] = diag

    if not seed_results:
        logger.warning("No seed results collected for Oracle Diagnostic.")
        return {}

    # Aggregate across seeds
    routed_mses = [v["overall_routed_mse"] for v in seed_results.values()]
    oracle_mses = [v["overall_oracle_mse"] for v in seed_results.values()]
    gaps = [v["overall_routing_gap"] for v in seed_results.values()]

    summary = {
        "mean_routed_mse": float(sum(routed_mses) / len(routed_mses)),
        "mean_oracle_mse": float(sum(oracle_mses) / len(oracle_mses)),
        "mean_routing_gap": float(sum(gaps) / len(gaps)),
        "seeds": seed_results,
    }

    # Persist summary
    out_file = out_base / "phaseforge" / "stage2" / "oracle_diagnostic.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Persisted Oracle Diagnostic summary to %s", out_file)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_dev)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
    run_oracle_diagnostics(output_dir=args.output_dir, seeds=tuple(args.seeds), device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
