"""Abstract base trainer defining the training lifecycle."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from hydra.utils import instantiate

from phaseforge.models.base import BaseManipulationModel

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base class for all training loops.

    Defines a standard lifecycle:
    fit -> [on_train_start] -> loop epochs -> [on_epoch_start] -> train_epoch -> 
    validate -> [on_epoch_end] -> end loop -> [on_train_end].

    Subclasses must implement _compute_loss.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: BaseManipulationModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
    ) -> None:
        self.cfg = cfg
        self.train_cfg = cfg.train
        
        self.device = torch.device(cfg.project.get("device", "cuda"))
        self.model = model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.epochs = self.train_cfg.epochs
        self.grad_clip_norm = self.train_cfg.grad_clip_norm
        self.log_every_n_steps = self.train_cfg.log_every_n_steps
        
        self.optimizer = instantiate(self.train_cfg.optimizer, params=self.model.parameters())
        self.scheduler = instantiate(self.train_cfg.scheduler, optimizer=self.optimizer)
        
        self.callbacks: list[Any] = []
        self.current_epoch = 0
        self.global_step = 0

    def add_callback(self, callback: Any) -> None:
        """Add a lifecycle callback."""
        self.callbacks.append(callback)

    def _trigger_callbacks(self, hook: str, **kwargs: Any) -> None:
        """Trigger a specific hook on all registered callbacks."""
        for cb in self.callbacks:
            method = getattr(cb, hook, None)
            if method:
                method(trainer=self, **kwargs)

    @abstractmethod
    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the loss for a single batch.

        Returns:
            total_loss: The scalar loss tensor to backpropagate.
            metrics: A dictionary of metric floats to log.
        """

    def fit(self) -> None:
        """Execute the full training loop."""
        logger.info(f"Starting training for {self.epochs} epochs on {self.device}.")
        self._trigger_callbacks("on_train_start")

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            self._trigger_callbacks("on_epoch_start")

            self._train_epoch()
            val_metrics = self._validate()

            self._trigger_callbacks("on_epoch_end", val_metrics=val_metrics)
            
            # Step the scheduler at the end of the epoch
            if self.scheduler:
                self.scheduler.step()

        self._trigger_callbacks("on_train_end")
        logger.info("Training complete.")

    def _train_epoch(self) -> None:
        self.model.train()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            self.optimizer.zero_grad()
            
            # Subclasses implement the specific loss logic
            loss, metrics = self._compute_loss(batch)
            
            loss.backward()
            
            if self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                
            self.optimizer.step()
            self.global_step += 1
            
            if self.global_step % self.log_every_n_steps == 0:
                self._trigger_callbacks("on_train_step", step=self.global_step, metrics=metrics)

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        agg_metrics: dict[str, float] = {}
        num_batches = 0
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # We don't backprop, just compute losses for logging
            _, metrics = self._compute_loss(batch)
            
            for k, v in metrics.items():
                agg_metrics[k] = agg_metrics.get(k, 0.0) + v
            num_batches += 1
            
        # Average
        if num_batches > 0:
            agg_metrics = {k: v / num_batches for k, v in agg_metrics.items()}
            
        return agg_metrics
