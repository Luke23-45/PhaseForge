"""Shared helpers for experiment scripts that need to load the trained model
and iterate the validation set offline (A3/A4/A5/B2/C1).

These scripts use the same Hydra compose path as the training CLI, but
instead of running training they instantiate the model, load a checkpoint
and run inference on the validation data with no grad.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_HYDRA_INITIALIZED = False


def compose_cfg(overrides, config_name: str = "main"):
    """Hydra compose. ``overrides`` is a list of dotted-key=val strings."""
    from hydra import initialize_config_dir

    global _HYDRA_INITIALIZED
    if not _HYDRA_INITIALIZED:
        initialize_config_dir(
            config_dir=str(PROJECT_ROOT / "phaseforge/config"),
            version_base=None,
            job_name="surgical_diagnostic",
        )
        _HYDRA_INITIALIZED = True
    from hydra import compose

    return compose(config_name=config_name, overrides=list(overrides))


def build_model_and_load(ckpt_path: Path, device, overrides=()):
    """Instantiate the phaseforge model, load the checkpoint, return ``(model, cfg)``."""
    from phaseforge.cli import _load_state_dict_checked
    from phaseforge.utils.registry import build_model

    cfg = compose_cfg(list(overrides))
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _load_state_dict_checked(model, ckpt["model_state_dict"], "eval")
    model.to(device).eval()
    return model, cfg


def build_val_loader(cfg):
    from phaseforge.utils.registry import build_data_pipeline

    loaders = build_data_pipeline(cfg).run()
    val_loader = loaders.get("val")
    if val_loader is None:
        raise RuntimeError("validation loader is empty")
    return val_loader


def iter_val_batches(model, val_loader, device, max_batches: int | None = None):
    """Yield (latent, action, phase_true, task_id) tensors. ``model.encoder``
    is used so we get the learned latents, not raw states."""
    encoder = model.encoder
    head = model.phase_head
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            state = batch["state"].to(device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if state.dim() == 3:
                state = state.squeeze(1)
            latent = encoder(state)
            phase_true = batch.get("phase_gt_clean", batch["phase"]).to(device)
            if phase_true.dim() == 2:
                phase_true = phase_true.squeeze(1)
            phase_logits = head(latent)
            yield {
                "latent": latent.detach().cpu(),
                "action": batch["action"].to(device).detach().cpu(),
                "phase_true": phase_true.detach().cpu(),
                "phase_pred": phase_logits.argmax(dim=-1).detach().cpu(),
                "task_id": batch["task_id"].detach().cpu(),
            }


def expert_outputs(model, latent: torch.Tensor) -> torch.Tensor:
    """Per-expert action outputs. Shape ``(B, E, A)``."""
    outs = []
    for expert in model.moe_layer.experts:
        outs.append(expert(latent))
    return torch.stack(outs, dim=1)


def gate_probs(model, latent: torch.Tensor) -> torch.Tensor:
    """Full softmax over the router logits. Shape ``(B, E)``."""
    logits = model.moe_layer.router.gate_linear(latent)
    return torch.softmax(logits, dim=-1)


def read_run_dir(run_dir: Path) -> dict:
    """Return ``{run_id, training_seed, tag, model, ...}`` from ``run_meta.json``."""
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def resolve_run_dir(seed_dir: Path, model_name: str | None = None, seed: int | None = None, tag: str | None = None, method_name: str | None = None) -> Path:
    """Resolve the timestamped run directory under ``seed_dir``.

    Prefers a run whose ``run_meta.json`` matches (model_name, seed, tag,
    method). Falls back to the single subdirectory when there is exactly one
    (the typical layout after a per-seed sweep run).
    """
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"no such directory: {seed_dir}")
    subdirs = [p for p in seed_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"no run subdirectory under {seed_dir}")
    if len(subdirs) == 1:
        meta = read_run_dir(subdirs[0])
        if method_name is not None and meta.get("method") != method_name:
            raise FileNotFoundError(
                f"single run under {seed_dir} is method={meta.get('method')!r}, "
                f"expected {method_name!r}"
            )
        if model_name is not None and meta.get("model_name") != model_name:
            raise FileNotFoundError(
                f"single run under {seed_dir} is model={meta.get('model_name')!r}, "
                f"expected {model_name!r}"
            )
        return subdirs[0]
    for d in subdirs:
        meta = read_run_dir(d)
        if model_name is not None and meta.get("model_name") != model_name:
            continue
        if seed is not None and meta.get("seed") != seed:
            continue
        if tag is not None and meta.get("tag") != tag:
            continue
        if method_name is not None and meta.get("method") != method_name:
            continue
        return d
    raise FileNotFoundError(
        f"no run under {seed_dir} matching model={model_name} seed={seed} "
        f"tag={tag} method={method_name}"
    )


def stage2_run_dir(outputs: Path, seed: int) -> Path:
    """Resolve the *Wave A1 sweep* stage-2 run for a seed.

    The sweep trains with ``project.method=phaseforge``; the Wave B/A3 cells
    (four_way_init, ablation_grid, validation_bank) reuse the same output
    tree with different method names, so the method filter keeps this from
    picking a sibling run.
    """
    return resolve_run_dir(
        outputs / "phaseforge" / "stage2" / f"seed{seed}",
        model_name="phaseforge",
        seed=seed,
        tag=None,
        method_name="phaseforge",
    )


def stage2_ckpt(outputs: Path, seed: int, epoch: int | None = None) -> Path:
    """Best or per-epoch checkpoint of the phaseforge stage-2 run for a seed."""
    run_dir = stage2_run_dir(outputs, seed)
    if epoch is None:
        return run_dir / "checkpoints" / "checkpoint_best.pt"
    return run_dir / "checkpoints" / f"checkpoint_epoch_{epoch:04d}.pt"


def device_from(arg: str = "auto"):
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)
