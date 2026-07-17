"""Early stopping callback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import math

from phaseforge.trains.callbacks.base import Callback

if TYPE_CHECKING:
    from phaseforge.trains.loops.base import BaseTrainer

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(Callback):
    """Stops training if a monitored metric stops improving."""

    def __init__(self, monitor: str = "val/loss_total", mode: str = "min", patience: int = 10, min_delta: float = 0.0) -> None:
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        
        self.wait_count = 0
        self.best_score = math.inf if mode == "min" else -math.inf

    def on_epoch_end(self, trainer: BaseTrainer, val_metrics: dict[str, float]) -> None:
        """Check if we should stop training."""
        if self.monitor not in val_metrics:
            logger.warning(
                f"Early stopping monitor '{self.monitor}' not found in val_metrics. "
                f"Available metrics: {list(val_metrics.keys())}"
            )
            return

        current_score = val_metrics[self.monitor]
        
        # Check for improvement
        improved = False
        if self.mode == "min":
            if current_score < self.best_score - self.min_delta:
                improved = True
        else:
            if current_score > self.best_score + self.min_delta:
                improved = True
                
        if improved:
            self.best_score = current_score
            self.wait_count = 0
        else:
            self.wait_count += 1
            logger.info(f"EarlyStopping: {self.monitor} did not improve. Patience: {self.wait_count}/{self.patience}")
            
            if self.wait_count >= self.patience:
                logger.info(f"EarlyStopping: Patience of {self.patience} reached. Signaling trainer to stop.")
                trainer.should_stop = True
