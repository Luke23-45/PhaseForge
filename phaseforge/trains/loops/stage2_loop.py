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

#: Cached 0.0 scalar per device. The aux-loss defaults below are requested
#: every training step; constructing a fresh device tensor each time is a
#: small per-step allocation × tens of thousands of steps per run. The cached
#: tensor is a read-only constant (value and dtype identical to
#: ``torch.tensor(0.0)``) — consumers only read or multiply it.
_ZERO_SCALARS: dict[torch.device, torch.Tensor] = {}


def _zero_scalar(device: torch.device) -> torch.Tensor:
    zero = _ZERO_SCALARS.get(device)
    if zero is None:
        zero = torch.zeros((), device=device)
        _ZERO_SCALARS[device] = zero
    return zero


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
        """Override fit to handle encoder freezing and fine-tuning LR scaling."""
        from hydra.utils import instantiate

        # Precedence: per-model models.freeze_encoder wins when key is present;
        # train.freeze_encoder otherwise; default True.
        models_cfg = (
            self.cfg.get("models")
            if hasattr(self, "cfg") and self.cfg is not None
            else None
        )
        if (
            models_cfg is not None
            and "freeze_encoder" in models_cfg
            and models_cfg.get("freeze_encoder") is not None
        ):
            freeze = bool(models_cfg.get("freeze_encoder"))
        elif self.train_cfg.get("freeze_encoder") is not None:
            freeze = bool(self.train_cfg.get("freeze_encoder"))
        else:
            freeze = True

        if freeze:
            if hasattr(self.model, "freeze_encoder"):
                self.model.freeze_encoder()
        elif hasattr(self.model, "unfreeze_encoder"):
            self.model.unfreeze_encoder()

        # Precedence: models.encoder_lr_scale wins if present, else
        # train.encoder_lr_scale, else 1.0.
        if (
            models_cfg is not None
            and "encoder_lr_scale" in models_cfg
            and models_cfg.get("encoder_lr_scale") is not None
        ):
            encoder_lr_scale = float(models_cfg.get("encoder_lr_scale"))
        elif self.train_cfg.get("encoder_lr_scale") is not None:
            encoder_lr_scale = float(self.train_cfg.get("encoder_lr_scale"))
        else:
            encoder_lr_scale = 1.0
        if not freeze and encoder_lr_scale != 1.0 and hasattr(self.model, "encoder"):
            base_lr = float(self.train_cfg.optimizer.get("lr", 1e-4))
            encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
            other_params = [
                p for n, p in self.model.named_parameters()
                if not n.startswith("encoder.") and p.requires_grad
            ]
            param_groups = [
                {"params": other_params, "lr": base_lr},
                {"params": encoder_params, "lr": base_lr * encoder_lr_scale},
            ]
            self.optimizer = instantiate(
                self.train_cfg.optimizer, params=param_groups, _convert_="all"
            )
            logger.info(
                f"Stage 2 Fine-Tuning: Encoder trained with scaled LR "
                f"{base_lr * encoder_lr_scale:.2e} (scale={encoder_lr_scale}), "
                f"other parameters at base LR {base_lr:.2e}."
            )
        else:
            active_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = instantiate(self.train_cfg.optimizer, params=active_params)

        self.scheduler = instantiate(self.train_cfg.scheduler, optimizer=self.optimizer)

        active_params = [p for p in self.model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in active_params)
        logger.info(f"Stage 2 initialized. Trainable parameters: {n_params}")

        super().fit()

    def _teacher_kl_coeff(self) -> float:
        """V2-D TGR schedule: λ0 for the first half, then linear anneal to 0.

        The teacher signal shapes the router early and is faded out so the
        learned gate takes over in the second half. ``lambda0 <= 0`` (the
        default) disables the schedule entirely.
        """
        tcfg = self.train_cfg.get("teacher_routing") or {}
        lambda0 = float(tcfg.get("lambda0", 0.0))
        if lambda0 <= 0.0:
            return 0.0
        total = max(1, int(self.train_cfg.get("epochs", 1)))
        t = min(max(float(self.current_epoch), 0.0), float(total)) / total
        if t <= 0.5:
            return lambda0
        return lambda0 * 2.0 * (1.0 - t)

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
        balance_loss = out.aux_losses.get("balance", _zero_scalar(self.device))

        # V2-C stickiness loss: -log p_t[top1_{t-1}], emitted by the router
        # when use_history=True. train.sticky_coeff scales it; 0.0 keeps the
        # loss bit-identical to the protocol (0.0 * x == 0.0).
        sticky_loss = out.aux_losses.get("sticky", _zero_scalar(self.device))
        sticky_coeff = float(self.train_cfg.get("sticky_coeff", 0.0))

        # V2-D teacher KL routing. Single source of truth is
        # models.teacher_routing.enabled (the same block the model's forward
        # reads to emit phase_logits in Stage 2); the trainer additionally
        # requires phase_logits present, so the teacher_forced cell (which
        # emits them through its own label-free eval path) never receives the
        # teacher KL. T = M^T softmax(phase_logits), detached; λ(t) = λ0 for
        # the first half of training, then linear anneal to 0.
        teacher_kl = _zero_scalar(self.device)
        teacher_lambda = 0.0
        models_cfg = self.cfg.get("models") or {}
        teacher_enabled = bool(
            (models_cfg.get("teacher_routing") or {}).get("enabled", False)
        )
        if (
            teacher_enabled
            and out.phase_logits is not None
            and out.gate_logits is not None
        ):
            require_soft_mapping = getattr(self.model, "require_soft_mapping", None)
            if require_soft_mapping is None:
                raise RuntimeError(
                    "models.teacher_routing.enabled=true but the model has no "
                    "require_soft_mapping() accessor — teacher KL needs the "
                    "soft phase->expert mapping M."
                )
            mapping = require_soft_mapping().detach()
            phase_probs = F.softmax(out.phase_logits, dim=-1)
            teacher_target = torch.einsum("pe,bp->be", mapping, phase_probs).detach()
            teacher_kl = F.kl_div(
                F.log_softmax(out.gate_logits, dim=-1),
                teacher_target,
                reduction="batchmean",
            )
            teacher_lambda = self._teacher_kl_coeff()

        # Total Loss
        total_loss = (
            action_loss
            + balance_loss
            + sticky_coeff * sticky_loss
            + teacher_lambda * teacher_kl
        )

        metrics = {
            # Defer .item() until the trainer logs/aggregates the metric so
            # the hot training loop does not synchronize CUDA every batch.
            "loss_total": total_loss.detach(),
            "loss_action": action_loss.detach(),
            "loss_balance": balance_loss.detach(),
            "loss_sticky": (sticky_coeff * sticky_loss).detach(),
            "loss_teacher_kl": (teacher_lambda * teacher_kl).detach(),
            "teacher_lambda": teacher_lambda,
        }

        return total_loss, metrics

    @torch.inference_mode()
    def _validate(self) -> dict[str, float]:
        """Validation plus per-epoch routing diagnostics (C3).

        Reuses the forward outputs for both the loss metrics and the routing
        metrics so validation stays a single forward pass per batch.

        Loss metrics are aggregated sample-weighted (see
        :meth:`BaseTrainer._batch_sample_count`) with on-device float64
        accumulation — bit-identical to the former Python-float arithmetic,
        materialized once at the epoch boundary instead of one host/device
        sync per tensor metric per batch. Routing tensors (expert indices,
        gate logits, phases, trajectory keys) are likewise collected on-device
        and moved to the host with a single ``torch.cat(...).cpu()`` per
        epoch.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        agg_sums: dict[str, torch.Tensor] = {}
        total_samples = 0
        self._val_routing_acc.reset()

        expert_indices_all: list[torch.Tensor] = []
        phases_all: list[torch.Tensor] = []
        gate_logits_all: list[torch.Tensor] = []
        traj_id_all: list[torch.Tensor] = []
        traj_pos_all: list[torch.Tensor] = []

        for batch in self.val_loader:
            batch = self._move_batch(batch)

            out = self.model(batch)
            _, metrics = self._compute_loss(batch, out=out)

            # Routing accuracy against GT phases when the model emits phase_logits
            # in Stage 2: the teacher_forced cell (label-free eval path) and
            # the V2-D teacher path both qualify.
            if out.phase_logits is not None:
                self._val_routing_acc.update(
                    out.phase_logits, batch["phase"], mask=batch.get("padding_mask")
                )

            n = self._batch_sample_count(batch)
            if n == 0:
                continue
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    contribution = v.detach().to(device=self.device, dtype=torch.float64)
                else:
                    contribution = torch.tensor(
                        float(v), device=self.device, dtype=torch.float64
                    )
                acc = agg_sums.get(k)
                agg_sums[k] = contribution * n if acc is None else acc + contribution * n
            total_samples += n

            if out.expert_indices is not None:
                expert_indices_all.append(out.expert_indices.detach())
                phases_all.append(batch["phase"])
                if "trajectory_id" in batch:
                    traj_id_all.append(batch["trajectory_id"])
                    traj_pos_all.append(batch["trajectory_position"])
            if out.gate_logits is not None:
                gate_logits_all.append(out.gate_logits.detach())

        if total_samples == 0:
            return {}
        agg_metrics = {k: float((v / total_samples).item()) for k, v in agg_sums.items()}

        # Routing diagnostics: balance-vs-specialization trajectory (C3).
        if expert_indices_all:
            expert_indices = torch.cat(expert_indices_all, dim=0).cpu()
            phases = torch.cat(phases_all, dim=0).cpu()

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
            gate_logits = torch.cat(gate_logits_all, dim=0).cpu()
            agg_metrics["val/routing_entropy"] = routing_stability.routing_entropy(
                gate_logits, normalize=True
            ).item()

        # V2-C: how often the top-1 expert changes between adjacent
        # in-trajectory steps (0.0 = perfectly sticky, 1.0 = flips every
        # step). Absent when the validation batches carry no trajectory keys.
        if traj_id_all:
            agg_metrics["val/routing_switch_rate"] = (
                routing_stability.routing_switch_rate(
                    expert_indices[:, 0],
                    torch.cat(traj_id_all, dim=0).cpu(),
                    torch.cat(traj_pos_all, dim=0).cpu(),
                )
            )

        if self._val_routing_acc.has_data:
            acc, balanced = self._val_routing_acc.compute()
            agg_metrics["val/routing_accuracy"] = acc
            agg_metrics["val/routing_balanced_accuracy"] = balanced

        logger.info(
            "Epoch %d routing diagnostics: NMI=%.4f topk_balance=%.4f "
            "top1_balance=%.4f topk_collapse=%.4f top1_collapse=%.4f entropy=%.4f "
            "switch_rate=%.4f",
            self.current_epoch,
            agg_metrics.get("val/phase_expert_nmi", float("nan")),
            agg_metrics.get("val/topk_balance_score", float("nan")),
            agg_metrics.get("val/top1_balance_score", float("nan")),
            agg_metrics.get("val/topk_collapse_rate", float("nan")),
            agg_metrics.get("val/top1_collapse_rate", float("nan")),
            agg_metrics.get("val/routing_entropy", float("nan")),
            agg_metrics.get("val/routing_switch_rate", float("nan")),
        )

        return agg_metrics
