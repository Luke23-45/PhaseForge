"""Command line interfaces for training and evaluation.

Entry points:
    phaseforge-train: Runs the training loop (Stage 1 or Stage 2).
    phaseforge-eval: Runs the evaluation loop.
"""

from __future__ import annotations

import logging

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from phaseforge.trains.callbacks.checkpointing import CheckpointCallback
from phaseforge.trains.callbacks.early_stopping import EarlyStoppingCallback
from phaseforge.trains.callbacks.metric_tracker import MetricTrackerCallback
from phaseforge.trains.callbacks.wandb_logger import WandbLoggerCallback
from phaseforge.utils.config import find_latest_checkpoint, get_output_dir, write_run_meta
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
        # Load Stage 1 checkpoint and bootstrap
        ckpt_path = cfg.train.get("stage1_ckpt_path")
        if not ckpt_path:
            model_name = getattr(cfg.models, "name", cfg.models._target_.split(".")[-1])
            auto_ckpt = find_latest_checkpoint(model_name, stage=1, base=cfg.project.output_dir)
            if auto_ckpt is not None:
                ckpt_path = str(auto_ckpt)
                logger.info(f"Auto-detected Stage 1 checkpoint: {ckpt_path}")
            else:
                raise ValueError(
                    "train.stage1_ckpt_path must be provided for Stage 2 training. "
                    f"No checkpoint auto-detected for model '{model_name}' stage 1."
                )

        logger.info(f"Loading Stage 1 checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # We load strict=False because the Stage 2 model has a MoELayer that was not
        # present or trained in Stage 1. We just want the encoder and action/phase heads.
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        # Execute the core bootstrapping algorithm
        if hasattr(model, "bootstrap_moe"):
            model.bootstrap_moe(dataloader=train_loader, device=cfg.project.get("device", "cuda"))
        else:
            logger.info("Model does not have bootstrap_moe(); assuming it's a standard baseline.")
            model.stage = 2

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
    """Main evaluation entry point."""
    set_seed(cfg.project.seed)
    output_dir = get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    write_run_meta(output_dir, cfg)

    logger.info(f"Evaluation output directory: {output_dir}")
    logger.info("Evaluation pipeline not fully implemented yet.")
    # TODO: Implement evaluate() invoking the OfflineEvaluator
