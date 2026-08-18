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

    def _phase_distill_weight(self) -> float:
        """Effective κ(t) for the phase-distillation warmup (V2).

        ``train.phase_distill`` (disabled by default):
        ``{enabled, kappa0, tau, anneal_epochs}``. The KL term
        ``κ(t)·E[KL(g(z) ‖ softmax(F(z)/τ))]`` shapes the router toward the
        frozen stage-1 phase predictor during the first ``anneal_epochs``
        epochs and is then released so the action loss takes over.
        """
        cfg = self.train_cfg.get("phase_distill") or {}
        if not cfg.get("enabled", False):
            return 0.0
        anneal_epochs = max(1, int(cfg.get("anneal_epochs", 40)))
        if self.current_epoch >= anneal_epochs:
            return 0.0
        kappa0 = float(cfg.get("kappa0", 1.0))
        t = min(max(float(self.current_epoch), 0.0), float(anneal_epochs))
        return kappa0 * (1.0 - t / float(anneal_epochs))

    @staticmethod
    def _phase_conditional_balance(
        gate_logits: torch.Tensor,
        phases: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        """Within-phase Switch balance loss (V4).

        ``L = Σ_p (E / |D_p|) Σ_i f_i^p · p_i^p`` where f_i^p is the fraction
        of phase-p tokens whose top-1 expert is i and p_i^p is the mean gate
        probability over phase-p tokens. Global balance (Switch) pushes every
        token's gate toward the uniform 1/E — the uniformizing force behind
        pseudo-balancing (docs/research/phase_utilization_design.md Lemma 2).
        Within-phase balance keeps utilization even inside each phase while
        allowing phases to commit to small expert subsets.

        Args:
            gate_logits: (B, E) raw gate logits.
            phases: (B,) integer phase labels.
            num_experts: E.

        Returns:
            Scalar loss.
        """
        probs = torch.softmax(gate_logits, dim=-1)
        top1 = gate_logits.argmax(dim=-1)
        B, E = gate_logits.shape
        loss = torch.zeros((), device=gate_logits.device)
        for p in torch.unique(phases):
            mask = phases == p
            n = int(mask.sum().item())
            if n == 0:
                continue
            p_probs = probs[mask]  # (n, E)
            p_top1 = top1[mask]  # (n,)
            f = torch.zeros((E,), device=gate_logits.device)
            if n > 0:
                f.scatter_add_(
                    0, p_top1, torch.ones_like(p_top1, dtype=gate_logits.dtype)
                )
                f = f / float(n)
            p_mean = p_probs.mean(dim=0)
            loss = loss + float(E) * (f * p_mean).sum() * (float(n) / float(B))
        return loss

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

        # Phase-conditional balance (V4): within-phase Switch balance
        # replaces the global uniformizing force. Opt-in via
        # train.phase_balance_coeff > 0 (the V4 cell also sets the router's
        # balance_coeff to 0 so the two forces do not fight).
        phase_balance_coeff = float(self.train_cfg.get("phase_balance_coeff", 0.0))
        if phase_balance_coeff > 0.0 and out.gate_logits is not None:
            phases = batch["phase"]
            if mask is not None:
                phases = phases[mask]
                gate_logits = out.gate_logits[mask]
            else:
                gate_logits = out.gate_logits
            phase_balance = self._phase_conditional_balance(
                gate_logits, phases, gate_logits.size(-1)
            )
            if phase_balance_coeff != 0.0:
                total_loss = total_loss + phase_balance_coeff * phase_balance
                metrics["loss_phase_balance"] = (phase_balance_coeff * phase_balance).detach()

        # Phase distillation warmup (V2): the frozen stage-1 phase predictor
        # teaches the router (KL), annealed to zero. Requires
        # models.emit_phase_logits=true so out.phase_logits is available.
        distill_kappa = self._phase_distill_weight()
        if distill_kappa > 0.0 and out.phase_logits is not None and out.gate_logits is not None:
            tau = float((self.train_cfg.get("phase_distill") or {}).get("tau", 1.0))
            teacher = torch.softmax(out.phase_logits / tau, dim=-1).detach()
            log_router = torch.log_softmax(out.gate_logits, dim=-1)
            kl = (teacher * (teacher.clamp_min(1e-9).log() - log_router)).sum(dim=-1)
            if mask is not None:
                kl = kl[mask].mean()
            else:
                kl = kl.mean()
            distill_loss = distill_kappa * kl
            total_loss = total_loss + distill_loss
            metrics["loss_phase_distill"] = distill_loss.detach()

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
