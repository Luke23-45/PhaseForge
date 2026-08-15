"""Stage 2 Trainer: Bootstrapped MoE specialization."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from phaseforge.evaluations.metrics import expert_utilization, phase_alignment, routing_stability
from phaseforge.models.base import ModelOutput
from phaseforge.trains.loops.base import (
    BaseTrainer,
    MetricValue,
    _PhaseAccumulator,
)

logger = logging.getLogger(__name__)


class Stage2Trainer(BaseTrainer):
    """Trainer for Stage 2.

    Computes: L_total = L_action + β_balance * L_balance
    Action loss is MSE. Balance loss is auxiliary.
    Optionally freezes the encoder.

    Every epoch also logs the routing diagnostics (C3: balance-vs-NMI
    trajectory) over the validation set — ``val/phase_expert_nmi``,
    ``val/topk_balance_score``, ``val/topk_collapse_rate``,
    ``val/top1_balance_score``, ``val/top1_collapse_rate``,
    ``val/routing_entropy`` —
    so pseudo-balancing (balance high while NMI collapses to 0) is visible
    at training time instead of only in offline evaluation.

    For the ``teacher_forced`` cell, the label-free routing accuracy (micro
    and macro/balanced) is computed over the SAME inference path that
    selects experts by the frozen phase head (``out.phase_logits`` is only
    non-None for that cell in Stage 2) — never from the GT-routed training
    path (final specification §4.4).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._val_routing_acc = _PhaseAccumulator(device=self.device)

    def fit(self) -> None:
        """Override fit to handle encoder freezing before the loop starts."""
        if self.train_cfg.freeze_encoder:
            self.model.freeze_encoder()

        # Re-initialize optimizer because freezing might have changed requires_grad
        from hydra.utils import instantiate

        active_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = instantiate(self.train_cfg.optimizer, params=active_params)
        self.scheduler = instantiate(self.train_cfg.scheduler, optimizer=self.optimizer)

        n_params = sum(p.numel() for p in active_params)
        logger.info(f"Stage 2 initialized. Trainable parameters: {n_params}")

        super().fit()

    def _compute_loss(
        self, batch: dict[str, torch.Tensor], out: ModelOutput | None = None
    ) -> tuple[torch.Tensor, dict[str, MetricValue]]:
        # Forward pass (reuse an existing output when provided, e.g. from the
        # validation loop, to avoid a double forward).
        if out is None:
            out = self.model(batch)

        # Ground truths
        target_action = batch["action"]  # (B, A) or (B, T, A)
        mask = batch.get("padding_mask")  # (B, T) boolean or None

        # Action Loss (MSE)
        if mask is not None:
            action_loss = F.mse_loss(out.action_pred, target_action, reduction="none")
            action_loss = action_loss[mask].mean()
        else:
            action_loss = F.mse_loss(out.action_pred, target_action)

        # Balance Loss
        balance_loss = out.aux_losses.get("balance", torch.tensor(0.0, device=self.device))

        # Total Loss
        total_loss = action_loss + balance_loss

        metrics = {
            # Defer .item() until the trainer logs/aggregates the metric so
            # the hot training loop does not synchronize CUDA every batch.
            "loss_total": total_loss.detach(),
            "loss_action": action_loss.detach(),
            "loss_balance": balance_loss.detach(),
        }

        return total_loss, metrics

    @torch.inference_mode()
    def _validate(self) -> dict[str, float]:
        """Validation plus per-epoch routing diagnostics (C3).

        Reuses the forward outputs for both the loss metrics and the routing
        metrics so validation stays a single forward pass per batch.

        Loss metrics are aggregated sample-weighted (see
        :meth:`BaseTrainer._batch_sample_count`), so a short final validation
        batch does not weigh as much as a full one.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        agg_metrics: dict[str, float] = {}
        total_samples = 0
        self._val_routing_acc.reset()

        expert_indices_all: list[torch.Tensor] = []
        phases_all: list[torch.Tensor] = []
        gate_logits_all: list[torch.Tensor] = []

        for batch in self.val_loader:
            batch = self._move_batch(batch)

            out = self.model(batch)
            _, metrics = self._compute_loss(batch, out=out)

            # teacher_forced only: ``phase_logits`` are emitted in the
            # label-free eval path (frozen phase head selects the expert).
            if out.phase_logits is not None:
                self._val_routing_acc.update(
                    out.phase_logits, batch["phase"], mask=batch.get("padding_mask")
                )

            n = self._batch_sample_count(batch)
            if n == 0:
                continue
            for k, v in metrics.items():
                agg_metrics[k] = agg_metrics.get(k, 0.0) + self._metric_to_float(v) * n
            total_samples += n

            if out.expert_indices is not None:
                expert_indices_all.append(out.expert_indices.detach().cpu())
                phases_all.append(batch["phase"].cpu())
            if out.gate_logits is not None:
                gate_logits_all.append(out.gate_logits.detach().cpu())

        if total_samples == 0:
            return {}
        agg_metrics = {k: v / total_samples for k, v in agg_metrics.items()}

        # Routing diagnostics: balance-vs-specialization trajectory (C3).
        if expert_indices_all:
            expert_indices = torch.cat(expert_indices_all, dim=0)
            phases = torch.cat(phases_all, dim=0)

            agg_metrics["val/phase_expert_nmi"] = phase_alignment.phase_expert_nmi(
                phases, expert_indices
            )

            # Use the CONFIGURED expert count (gate-logit width), never the
            # largest observed index: an unused expert must still count as
            # dead, otherwise collapse/balance look healthy while an expert
            # never fires. Note the balance/collapse diagnostics count all
            # top-k assignments, whereas the training balance loss uses
            # top-1 assignments (documented in TopKRouter._compute_balance_loss).
            if gate_logits_all:
                num_experts = gate_logits_all[0].size(-1)
            else:
                num_experts = int(expert_indices.max().item()) + 1
            topk_fractions = expert_utilization.expert_utilization(expert_indices, num_experts)
            top1_fractions = expert_utilization.expert_utilization_top1(expert_indices, num_experts)
            agg_metrics["val/topk_balance_score"] = expert_utilization.expert_utilization_balance(
                topk_fractions
            )
            agg_metrics["val/top1_balance_score"] = expert_utilization.expert_utilization_balance(
                top1_fractions
            )
            agg_metrics["val/topk_collapse_rate"] = expert_utilization.collapse_rate(topk_fractions)
            agg_metrics["val/top1_collapse_rate"] = expert_utilization.collapse_rate(top1_fractions)

        if gate_logits_all:
            gate_logits = torch.cat(gate_logits_all, dim=0)
            agg_metrics["val/routing_entropy"] = routing_stability.routing_entropy(
                gate_logits, normalize=True
            ).item()

        if self._val_routing_acc.has_data:
            acc, balanced = self._val_routing_acc.compute()
            agg_metrics["val/routing_accuracy"] = acc
            agg_metrics["val/routing_balanced_accuracy"] = balanced

        logger.info(
            "Epoch %d routing diagnostics: NMI=%.4f topk_balance=%.4f "
            "top1_balance=%.4f topk_collapse=%.4f top1_collapse=%.4f entropy=%.4f",
            self.current_epoch,
            agg_metrics.get("val/phase_expert_nmi", float("nan")),
            agg_metrics.get("val/topk_balance_score", float("nan")),
            agg_metrics.get("val/top1_balance_score", float("nan")),
            agg_metrics.get("val/topk_collapse_rate", float("nan")),
            agg_metrics.get("val/top1_collapse_rate", float("nan")),
            agg_metrics.get("val/routing_entropy", float("nan")),
        )

        return agg_metrics
