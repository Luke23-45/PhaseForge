"""V5 — Viterbi-decoded routing evaluation (offline, CPU).

Loads a Stage 2 checkpoint, regroups the validation split into trajectories,
and evaluates:

1. per-step phase accuracy: argmax of the frozen stage-1 phase head;
2. decoded phase accuracy: Viterbi (monotone empirical transition prior) on
   the phase-head logits;
3. router phase accuracy: the trained router's top-1 expert mapped to a
   phase through the measured affinity matrix;
4. decoded router accuracy: Viterbi directly on the gate logits with the
   affinity-induced expert transition prior;
5. routing agreement: fraction of steps where the decoded and learned
   top-1 experts agree;
6. action MSE under three routings: learned top-2, hard phase-decoded
   (expert = affinity top-1 of the decoded phase), hard router-decoded
   (expert = MAP expert) — GT action MSE each.

The transition prior is built from TRAINING trajectories (phase process
knowledge); emissions come from the checkpoint under test. The model is
rebuilt from the run's OWN ``resolved_config.yaml`` (variant structure
included) and the checkpoint is strict-loaded. No training happens.
Usage::

    uv run python scripts/viterbi_routing_eval.py \
        models=phaseforge train=stage2 project.seed=42 \
        train.stage1_ckpt_path=<stage2 run directory> \
        project.output_dir=outputs_local_train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from phaseforge.evaluations.decoding.viterbi import (
    decode_phase_sequence,
    decode_router_sequence,
)
from phaseforge.utils.registry import build_data_pipeline, build_model
from phaseforge.utils.seed import set_seed


def _patch_argparse() -> None:
    """Same hydra 1.3.5 / Python 3.14 workaround as scripts/local_run.py."""
    orig_expand = argparse.HelpFormatter._expand_help

    def _safe_expand(self, action):
        if not isinstance(action.help, str):
            action.help = str(action.help)
        return orig_expand(self, action)

    argparse.HelpFormatter._expand_help = _safe_expand


_patch_argparse()


def _accumulate_transition_counts(trajectories: list[dict]) -> torch.Tensor:
    """Count phase transitions per trajectory (no cross-trajectory edges)."""
    max_p = 0
    for traj in trajectories:
        max_p = max(max_p, int(traj["phase"].max().item()))
    counts = torch.zeros((max_p + 1, max_p + 1), dtype=torch.float64)
    for traj in trajectories:
        ph = traj["phase"]
        if ph.numel() >= 2:
            # bincount per trajectory (duplicate transitions must accumulate).
            flat = ph[:-1] * (max_p + 1) + ph[1:]
            counts += torch.bincount(
                flat, minlength=(max_p + 1) * (max_p + 1)
            ).reshape(max_p + 1, max_p + 1).to(torch.float64)
    row_sums = counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return (counts / row_sums).float()


@hydra.main(version_base="1.3", config_path="../phaseforge/config", config_name="main")
def main(cfg: DictConfig) -> None:
    run_dir = Path(cfg.train.get("stage1_ckpt_path") or "")
    resolved = run_dir / "resolved_config.yaml"
    if not resolved.exists():
        raise RuntimeError(
            "train.stage1_ckpt_path must be a Stage 2 RUN DIRECTORY containing "
            f"resolved_config.yaml (got {run_dir})."
        )
    run_cfg = OmegaConf.load(resolved)
    cfg = OmegaConf.merge(cfg, run_cfg)
    set_seed(cfg.project.seed)
    device = torch.device("cpu")
    cfg.project.device = "cpu"

    pipeline = build_data_pipeline(cfg)
    dataloaders = pipeline.run()
    val_loader = dataloaders.get("val") or dataloaders.get("test")
    if val_loader is None:
        raise RuntimeError("No validation split available.")
    val_trajectories = val_loader.dataset.trajectories

    # Build the model from the RUN's own resolved config so variant-specific
    # structure (router anchor / init / emit_phase_logits) is reproduced
    # exactly; the checkpoint is then STRICT-loaded (any mismatch is a bug).
    model = build_model(cfg)
    ckpt = torch.load(
        run_dir / "checkpoints" / "checkpoint_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch: missing {missing[:8]}, "
            f"unexpected {unexpected[:8]}"
        )
    if hasattr(model, "stage") and "stage" in ckpt:
        model.stage = ckpt["stage"]
    model.to(device)
    model.eval()

    train_loader = dataloaders.get("train")
    train_trajectories = train_loader.dataset.trajectories
    transition = _accumulate_transition_counts(train_trajectories)

    # Phase→expert affinity from the trained router over val tokens.
    num_experts = model.moe_layer.router.num_experts
    num_phases = model.phase_head.num_phases
    affinity_acc = torch.zeros((num_phases, num_experts))
    phase_counts = torch.zeros((num_phases,))
    with torch.inference_mode():
        for traj in val_trajectories:
            state = traj["state"].to(device)
            phase = traj["phase"].to(device)
            gate = model.moe_layer.router(model.encoder(state))
            probs = torch.softmax(gate.gate_logits, dim=-1)
            affinity_acc.index_add_(0, phase, probs)
            phase_counts.index_add_(0, phase, torch.ones_like(phase, dtype=torch.float))
    affinity = affinity_acc / phase_counts.unsqueeze(1).clamp_min(1.0)

    # Per-trajectory evaluation.
    agg = {
        "phase_argmax_acc": 0.0,
        "phase_decoded_acc": 0.0,
        "router_phase_acc": 0.0,
        "router_decoded_acc": 0.0,
        "routing_agreement": 0.0,
        "mse_learned": 0.0,
        "mse_phase_decoded": 0.0,
        "mse_router_decoded": 0.0,
        "decoded_phase_run_length": 0.0,
        "steps": 0,
        "trajectories": 0,
    }
    expert_to_phase = affinity.argmax(dim=0)  # (E,) most-associated phase

    with torch.inference_mode():
        for traj in val_trajectories:
            state = traj["state"].to(device)
            phase = traj["phase"].to(device)
            target = traj["action"].to(device)
            T = state.shape[0]

            latent = model.encoder(state)
            phase_logits = model.phase_head(latent)
            gate = model.moe_layer.router(latent)
            gate_logits = gate.gate_logits

            # 1. per-step argmax of the phase head
            phase_argmax = phase_logits.argmax(dim=-1)

            # 2. Viterbi-decoded phase sequence
            phase_decoded = decode_phase_sequence(phase_logits, transition)

            # 3. router's implied phase (top-1 expert -> affinity phase)
            router_top1 = gate_logits.argmax(dim=-1)
            router_phase = expert_to_phase[router_top1]

            # 4. Viterbi-decoded router sequence -> implied phase
            router_decoded = decode_router_sequence(gate_logits, transition, affinity)
            router_decoded_phase = expert_to_phase[router_decoded]

            # 5. routing agreement (decoded top-1 vs learned top-1)
            agreement = (router_decoded == router_top1).float().mean().item()

            # 6. actions: learned vs hard phase-decoded vs hard router-decoded
            learned_action = model.moe_layer(latent).combined_output
            phase_expert = affinity[phase_decoded].argmax(dim=-1)  # (T,)
            phase_action = torch.stack(
                [model.moe_layer.experts[e](latent[i]) for i, e in enumerate(phase_expert)]
            )
            router_action = torch.stack(
                [model.moe_layer.experts[e](latent[i]) for i, e in enumerate(router_decoded)]
            )

            n = float(T)
            agg["phase_argmax_acc"] += (phase_argmax == phase).float().mean().item() * n
            agg["phase_decoded_acc"] += (phase_decoded == phase).float().mean().item() * n
            agg["router_phase_acc"] += (router_phase == phase).float().mean().item() * n
            agg["router_decoded_acc"] += (
                router_decoded_phase == phase
            ).float().mean().item() * n
            agg["routing_agreement"] += agreement * n
            agg["mse_learned"] += torch.nn.functional.mse_loss(learned_action, target).item() * n
            agg["mse_phase_decoded"] += (
                torch.nn.functional.mse_loss(phase_action, target).item() * n
            )
            agg["mse_router_decoded"] += (
                torch.nn.functional.mse_loss(router_action, target).item() * n
            )
            # segment coherence: mean run length of the decoded phase sequence
            runs = 1 + int((phase_decoded[1:] != phase_decoded[:-1]).sum().item())
            agg["decoded_phase_run_length"] += (float(T) / runs) * n
            agg["steps"] += n
            agg["trajectories"] += 1

    results = {
        k: (v / agg["steps"] if k not in ("steps", "trajectories") else v)
        for k, v in agg.items()
    }
    results["seed"] = cfg.project.seed
    results["run_dir"] = run_dir.name
    results["ckpt"] = "checkpoints/checkpoint_best.pt"

    print(json.dumps(results, indent=2, sort_keys=True))

    out_dir = Path(cfg.project.output_dir) / "v5_decoded_routing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"v5_seed{cfg.project.seed}_{run_dir.name}.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()