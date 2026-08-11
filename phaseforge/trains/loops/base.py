"""Abstract base trainer defining the training lifecycle."""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

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
        
        requested_device = str(cfg.project.get("device", "cuda"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "Device '%s' requested but CUDA is unavailable. Falling back to CPU.",
                requested_device,
            )
            requested_device = "cpu"
        self.device = torch.device(requested_device)
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
        self.should_stop = False
        # Checkpoint payload queued by resume(); applied at the start of the
        # next fit() so subclasses that re-create the optimizer (e.g. after
        # freezing) restore state into the final optimizer instance.
        self._resume_payload: dict[str, Any] | None = None

    def resume(self, checkpoint_path: str | Path) -> None:
        """Queue a checkpoint for restoration at the next :meth:`fit`.

        Restores model weights, optimizer/scheduler state, epoch, global
        step, RNG state (torch/numpy/random/cuda) and callback state so an
        interrupted run can resume from exactly where it stopped. Applied in
        :meth:`fit` (after any optimizer re-instantiation by subclasses).

        Args:
            checkpoint_path: Path to a checkpoint saved by the
                :class:`~phaseforge.trains.callbacks.checkpointing.CheckpointCallback`.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        required = {"model_state_dict", "optimizer_state_dict"}
        missing = required - set(ckpt.keys())
        if missing:
            raise ValueError(
                f"Resume checkpoint {path} is missing required keys: {sorted(missing)}"
            )
        self._resume_payload = ckpt
        logger.info(f"Queued checkpoint for resume: {path}")

    def _apply_resume_payload(self) -> None:
        """Restore the state recorded by :meth:`resume` into this trainer."""
        ckpt = self._resume_payload
        if ckpt is None:
            return
        self._resume_payload = None

        # Strict load: resuming into a different architecture must fail
        # loudly instead of silently leaving random weights.
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.current_epoch = int(ckpt.get("epoch", self.current_epoch))
        self.global_step = int(ckpt.get("global_step", self.global_step))
        self.should_stop = bool(ckpt.get("should_stop", self.should_stop))
        if hasattr(self.model, "stage") and ckpt.get("stage") is not None:
            setattr(self.model, "stage", int(ckpt["stage"]))

        rng = ckpt.get("rng_state")
        if isinstance(rng, dict):
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"].cpu())
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("random") is not None:
                random.setstate(rng["random"])
            cuda_states = rng.get("cuda")
            if cuda_states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cuda() for s in cuda_states])

        callbacks = ckpt.get("callbacks", {})
        for cb in self.callbacks:
            state = callbacks.get(type(cb).__name__)
            if state is not None and hasattr(cb, "load_state_dict"):
                cb.load_state_dict(state)

        logger.info(
            "Resumed training at epoch %d, global step %d.",
            self.current_epoch,
            self.global_step,
        )

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
    def _compute_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the loss for a single batch.

        Returns:
            total_loss: The scalar loss tensor to backpropagate.
            metrics: A dictionary of metric floats to log.
        """

    def fit(self) -> None:
        """Execute the full training loop."""
        # Restore resume state now: subclasses that re-create the optimizer
        # (e.g. after freezing the encoder) have done so by this point.
        self._apply_resume_payload()

        logger.info(f"Starting training for {self.epochs} epochs on {self.device}.")
        self._trigger_callbacks("on_train_start")

        for epoch in range(self.current_epoch + 1, self.epochs + 1):
            self.current_epoch = epoch
            self._trigger_callbacks("on_epoch_start")

            self._train_epoch()
            val_metrics = self._validate()

            self._trigger_callbacks("on_epoch_end", val_metrics=val_metrics)
            
            # Step the scheduler at the end of the epoch
            if self.scheduler:
                self.scheduler.step()

            if self.should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        self._trigger_callbacks("on_train_end")
        logger.info("Training complete.")

    def _train_epoch(self) -> None:
        self.model.train()
        
        use_pbar = self.train_cfg.get("rich_progressbar", False)
        
        if use_pbar:
            try:
                from tqdm.rich import tqdm
            except ImportError:
                from tqdm import tqdm
                
            pbar = tqdm(
                self.train_loader, 
                desc=f"Epoch {self.current_epoch}/{self.epochs} [Train]",
                leave=False
            )
            iterable = pbar
        else:
            pbar = None
            iterable = self.train_loader
            
        for batch_idx, batch in enumerate(iterable):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            self.optimizer.zero_grad()
            
            # Subclasses implement the specific loss logic
            loss, metrics = self._compute_loss(batch)
            
            # Fail fast on non-finite losses: a NaN/Inf loss would otherwise
            # corrupt the optimizer state and the checkpoint silently.
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss ({loss.item()}) at global step "
                    f"{self.global_step + 1}, epoch {self.current_epoch}. "
                    "Aborting to avoid corrupting the run."
                )

            loss.backward()

            if self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            # Fail fast on non-finite gradients (e.g. NaN from a poisoned
            # batch); check before the optimizer step.
            for name, param in self.model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(
                        f"Non-finite gradient in '{name}' at global step "
                        f"{self.global_step + 1}, epoch {self.current_epoch}. "
                        "Aborting to avoid corrupting the run."
                    )

            self.optimizer.step()
            self.global_step += 1
            
            if self.global_step % self.log_every_n_steps == 0:
                self._trigger_callbacks("on_train_step", step=self.global_step, metrics=metrics)
                
            if pbar is not None:
                postfix = {k: f"{v:.4f}" for k, v in metrics.items()}
                postfix["loss"] = f"{loss.item():.4f}"
                pbar.set_postfix(postfix)

    def _batch_sample_count(self, batch: dict[str, torch.Tensor]) -> int:
        """Number of valid samples a batch contributes to the losses.

        Uses the padding mask when present (``drop_last=False`` validation
        batches can be short), otherwise the batch size.
        """
        mask = batch.get("padding_mask")
        if mask is not None:
            return int(mask.sum().item())
        return batch["action"].shape[0]

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        agg_metrics: dict[str, float] = {}
        total_samples = 0
        
        use_pbar = self.train_cfg.get("rich_progressbar", False)
        
        if use_pbar:
            try:
                from tqdm.rich import tqdm
            except ImportError:
                from tqdm import tqdm
                
            pbar = tqdm(
                self.val_loader, 
                desc=f"Epoch {self.current_epoch}/{self.epochs} [Val]",
                leave=False
            )
            iterable = pbar
        else:
            pbar = None
            iterable = self.val_loader
            
        for batch in iterable:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # We don't backprop, just compute losses for logging
            _, metrics = self._compute_loss(batch)
            
            if pbar is not None:
                postfix = {k: f"{v:.4f}" for k, v in metrics.items()}
                pbar.set_postfix(postfix)
                
            # Weight per-batch metrics by the number of valid samples so the
            # final (short) validation batch does not get the same weight as a
            # full batch (validation uses drop_last=False).
            n = self._batch_sample_count(batch)
            if n == 0:
                continue
            for k, v in metrics.items():
                agg_metrics[k] = agg_metrics.get(k, 0.0) + v * n
            total_samples += n
            
        if total_samples > 0:
            agg_metrics = {k: v / total_samples for k, v in agg_metrics.items()}
            
        return agg_metrics
