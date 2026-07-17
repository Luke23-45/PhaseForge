"""Command line interfaces for training and evaluation.

Entry points:
    phaseforge-train: Runs the training loop (Stage 1 or Stage 2).
    phaseforge-eval: Runs the evaluation loop.
"""

from __future__ import annotations

import json
import logging

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from phaseforge.trains.callbacks.checkpointing import CheckpointCallback
from phaseforge.trains.callbacks.early_stopping import EarlyStoppingCallback
from phaseforge.trains.callbacks.metric_tracker import MetricTrackerCallback
from phaseforge.trains.callbacks.wandb_logger import WandbLoggerCallback
from phaseforge.utils.config import (
    find_latest_checkpoint,
    get_eval_output_dir,
    get_output_dir,
    resolve_checkpoint_source,
    write_run_meta,
)
from phaseforge.utils.registry import build_data_pipeline, build_model, build_trainer
from phaseforge.utils.seed import set_seed

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="config", config_name="main")
def train(cfg: DictConfig) -> None:
    """Main training entry point."""
    # 1. Setup
    set_seed(cfg.project.seed)
    output_dir = get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config and lightweight run metadata
    with open(output_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    write_run_meta(output_dir, cfg)

    logger.info(f"Output directory: {output_dir}")

    # 2. Init W&B
    if cfg.project.wandb.mode != "disabled":
        wandb.init(
            project=cfg.project.wandb.project,
            entity=cfg.project.wandb.entity,
            mode=cfg.project.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=str(output_dir),
        )

    # 3. Data Pipeline
    logger.info("Initializing Data Pipeline...")
    pipeline = build_data_pipeline(cfg)
    dataloaders = pipeline.run()
    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("val")

    if train_loader is None:
        raise RuntimeError("No training data found. Check split ratios and cache.")

    # 4. Model
    logger.info("Initializing Model...")
    model = build_model(cfg)

    stage = cfg.train.get("stage", 1)

    if stage == 2:
        ckpt_path = cfg.train.get("stage1_ckpt_path")

        if hasattr(model, "bootstrap_moe"):
            # Models with bootstrapping (PhaseBootstrappedMoE, WarmStartMoE)
            # need a Stage 1 checkpoint to initialise encoder + action_head.
            if not ckpt_path:
                model_name = getattr(cfg.models, "name", cfg.models._target_.split(".")[-1])
                source_model = resolve_checkpoint_source(model_name)
                auto_ckpt = find_latest_checkpoint(
                    source_model, stage=1, base=cfg.project.output_dir,
                    resolve_alias=False,
                )
                if auto_ckpt is not None:
                    ckpt_path = str(auto_ckpt)
                    logger.info(
                        f"Auto-detected Stage 1 checkpoint (from '{source_model}'): {ckpt_path}"
                    )
                else:
                    raise ValueError(
                        f"{type(model).__name__} requires a Stage 1 checkpoint. "
                        f"Set train.stage1_ckpt_path or ensure "
                        f"outputs/{source_model}/stage1/ has one."
                    )

            logger.info(f"Loading Stage 1 checkpoint from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)

            model.bootstrap_moe(dataloader=train_loader, device=cfg.project.get("device", "cuda"))
        else:
            # Models without bootstrapping (ScratchMoE, OraclePhaseMoE)
            # train from scratch — no checkpoint needed.
            logger.info(f"{type(model).__name__}: No bootstrapping. Training from scratch.")

    # 5. Trainer
    logger.info(f"Initializing Stage {stage} Trainer...")
    trainer = build_trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # 6. Callbacks
    trainer.add_callback(CheckpointCallback(
        output_dir=output_dir / "checkpoints",
        every_n_epochs=cfg.train.checkpoint.every_n_epochs,
        monitor=cfg.train.checkpoint.monitor,
        mode=cfg.train.checkpoint.mode,
        save_top_k=cfg.train.checkpoint.save_top_k,
    ))
    trainer.add_callback(MetricTrackerCallback())

    if hasattr(cfg.train, "early_stopping") or "early_stopping" in cfg.train:
        trainer.add_callback(EarlyStoppingCallback(
            monitor=cfg.train.early_stopping.monitor,
            mode=cfg.train.early_stopping.mode,
            patience=cfg.train.early_stopping.patience,
            min_delta=cfg.train.early_stopping.min_delta,
        ))

    if cfg.project.wandb.mode != "disabled":
        trainer.add_callback(WandbLoggerCallback())

    # 7. Go!
    trainer.fit()

    if wandb.run is not None:
        wandb.finish()


@hydra.main(version_base="1.3", config_path="config", config_name="main")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model on the validation/test set."""
    set_seed(cfg.project.seed)
    output_dir = get_eval_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    write_run_meta(output_dir, cfg)

    logger.info(f"Evaluation output directory: {output_dir}")

    # 1. Data Pipeline
    logger.info("Initializing Data Pipeline...")
    pipeline = build_data_pipeline(cfg)
    dataloaders = pipeline.run()
    val_loader = dataloaders.get("val") or dataloaders.get("test")
    if val_loader is None:
        raise RuntimeError("No validation/test data found for evaluation.")

    # 2. Model
    logger.info("Initializing Model...")
    model = build_model(cfg)

    # 3. Load checkpoint
    ckpt_path = cfg.train.get("stage1_ckpt_path")
    if ckpt_path:
        logger.info(f"Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        # Restore the stage attribute — it is a plain Python int, NOT in state_dict(),
        # so load_state_dict() leaves it at the __init__ default (1).
        if hasattr(model, "stage") and "stage" in ckpt:
            model.stage = ckpt["stage"]
    else:
        logger.warning(
            "No checkpoint provided (train.stage1_ckpt_path). "
            "Using randomly initialized model."
        )

    # 4. Run offline evaluation
    from phaseforge.evaluations.runners.offline_evaluator import OfflineEvaluator

    evaluator = OfflineEvaluator(cfg=cfg, model=model, dataloader=val_loader)
    results = evaluator.run()

    # 5. Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Evaluation complete:")
    for key, val in results.items():
        logger.info(f"  {key}: {val:.6f}")
    logger.info(f"Results saved to {results_path}")
