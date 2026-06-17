"""CheckpointCallback: Saves model weights periodically and tracks best models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch

from phaseforge.trains.callbacks.base import Callback

if TYPE_CHECKING:
    from phaseforge.trains.loops.base import BaseTrainer

logger = logging.getLogger(__name__)


class CheckpointCallback(Callback):
    """Saves checkpoints during training.

    Args:
        output_dir: Directory to save checkpoints.
        every_n_epochs: Save frequency.
        monitor: Metric to monitor for the 'best' checkpoint.
        mode: 'min' or 'max' for the monitored metric.
        save_top_k: How many best checkpoints to keep.
    """

    def __init__(
        self,
        output_dir: Path | str,
        every_n_epochs: int = 10,
        monitor: str = "val/loss_total",
        mode: str = "min",
        save_top_k: int = 1,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.every_n_epochs = every_n_epochs
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k

        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_ckpt_path: Path | None = None

    def on_epoch_end(self, trainer: BaseTrainer, val_metrics: dict[str, float]) -> None:
        epoch = trainer.current_epoch
        
        # Determine if this is a new best model
        # Metric keys might be passed as e.g., 'loss_total' instead of 'val/loss_total'
        # so we strip 'val/' if necessary.
        monitor_key = self.monitor.replace("val/", "")
        
        current_score = val_metrics.get(monitor_key)
        is_best = False
        
        if current_score is not None:
            if self.mode == "min" and current_score < self.best_score:
                self.best_score = current_score
                is_best = True
            elif self.mode == "max" and current_score > self.best_score:
                self.best_score = current_score
                is_best = True

        # Save logic
        if is_best:
            self._save(trainer, epoch, is_best=True)
            
        if epoch % self.every_n_epochs == 0:
            self._save(trainer, epoch, is_best=False)

    def _save(self, trainer: BaseTrainer, epoch: int, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "global_step": trainer.global_step,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler else None,
            "stage": trainer.model.stage if hasattr(trainer.model, "stage") else 1,
        }
        
        if is_best:
            path = self.output_dir / "checkpoint_best.pt"
            torch.save(state, path)
            self.best_ckpt_path = path
            logger.info(f"Saved new best checkpoint (epoch {epoch}) to {path.name}")
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            torch.save(state, path)
            logger.info(f"Saved periodic checkpoint to {path.name}")
