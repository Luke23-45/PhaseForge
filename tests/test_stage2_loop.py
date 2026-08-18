"""CPU-only tests for the Stage 2 trainer routing diagnostics (C3).

Verifies that every epoch's validation records the balance-vs-specialization
trajectory (``val/phase_expert_nmi``, ``val/topk_balance_score``,
``val/topk_collapse_rate``, ``val/top1_balance_score``,
``val/top1_collapse_rate``, ``val/routing_entropy``) and that validation reuses
the model forward outputs (no double forward pass).
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from phaseforge.models.base import ModelOutput
from phaseforge.trains.loops.stage2_loop import Stage2Trainer


class _DictDataset(Dataset):
    def __init__(self, num: int = 32, num_phases: int = 3, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.states = torch.randn(num, 4, generator=gen)
        self.actions = torch.randn(num, 2, generator=gen)
        self.phases = torch.randint(0, num_phases, (num,), generator=gen)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "phase": self.phases[idx],
        }


class CountingMoEModel(torch.nn.Module):
    """Deterministic fake MoE: routes every sample to its GT phase."""

    def __init__(self, num_experts: int = 3) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.forward_calls = 0
        # A real parameter so the trainer's optimizer has something to own.
        self.probe = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        self.forward_calls += 1
        phase = batch["phase"]
        B = phase.size(0)
        indices = phase.unsqueeze(-1)
        weights = torch.ones(B, 1)
        gate = torch.zeros(B, self.num_experts)
        gate.scatter_(1, indices, 100.0)
        return ModelOutput(
            action_pred=self.probe * torch.zeros(B, 2),
            phase_logits=None,
            routing_weights=weights,
            expert_indices=indices,
            gate_logits=gate,
            aux_losses={"balance": torch.tensor(0.0)},
        )

    def freeze_encoder(self) -> None:
        pass


class NaNMoEModel(CountingMoEModel):
    """Fake MoE with a poisoned forward: NaN action predictions."""

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        out = super().forward(batch)
        return ModelOutput(
            action_pred=torch.full_like(out.action_pred, float("nan")),
            phase_logits=None,
            routing_weights=out.routing_weights,
            expert_indices=out.expert_indices,
            gate_logits=out.gate_logits,
            aux_losses=out.aux_losses,
        )


class SqrtPoisonModel(torch.nn.Module):
    """Finite forward but a NaN gradient (``sqrt`` derivative at zero).

    ``sqrt(0)`` is 0 (finite loss) but its local derivative is infinite;
    multiplying by a zero constant makes the chain-rule product
    ``0 * inf = NaN``, so the gradient guard must catch it while the loss
    guard sees a perfectly finite loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        B = batch["action"].size(0)
        action_pred = torch.zeros(B, 2) + torch.sqrt(self.a) * 0.0
        return ModelOutput(
            action_pred=action_pred,
            phase_logits=None,
            routing_weights=None,
            expert_indices=None,
            gate_logits=None,
            aux_losses={"balance": torch.tensor(0.0)},
        )

    def freeze_encoder(self) -> None:
        pass


class PhaseEmittingMoEModel(torch.nn.Module):
    """Fake MoE with deterministic gate + phase-head logits (V2/V4 tests).

    Routes every sample one-hot to its GT phase and emits a one-hot phase
    predictor that always predicts phase 0.
    """

    def __init__(self, num_experts: int = 3) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.probe = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        phase = batch["phase"]
        B = phase.size(0)
        gate = torch.zeros(B, self.num_experts)
        gate.scatter_(1, phase.unsqueeze(-1), 100.0)
        phase_logits = torch.zeros(B, self.num_experts)
        phase_logits[:, 0] = 10.0
        return ModelOutput(
            action_pred=self.probe * torch.zeros(B, 2),
            phase_logits=phase_logits,
            routing_weights=torch.ones(B, 1),
            expert_indices=phase.unsqueeze(-1),
            gate_logits=gate,
            aux_losses={"balance": torch.tensor(0.0)},
        )

    def freeze_encoder(self) -> None:
        pass


def _make_trainer(model: torch.nn.Module, val_loader: DataLoader) -> Stage2Trainer:
    cfg = DictConfig(
        {
            "project": {"device": "cpu"},
            "train": {
                "epochs": 1,
                "grad_clip_norm": 0.0,
                "log_every_n_steps": 1,
                "freeze_encoder": False,
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "lr": 1.0e-4,
                },
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                    "T_max": 1,
                },
            },
        }
    )
    return Stage2Trainer(cfg=cfg, model=model, train_loader=val_loader, val_loader=val_loader)


def test_validate_reports_routing_diagnostics() -> None:
    model = CountingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(seed=1), batch_size=8))

    val_metrics = trainer._validate()

    # Perfect GT routing => perfect phase-expert alignment (NMI ~ 1.0) and
    # near-uniform usage => high balance score.
    assert "val/phase_expert_nmi" in val_metrics
    assert val_metrics["val/phase_expert_nmi"] > 0.99
    assert "val/topk_balance_score" in val_metrics
    assert val_metrics["val/topk_balance_score"] > 0.9
    assert "val/top1_balance_score" in val_metrics
    assert "val/topk_collapse_rate" in val_metrics
    assert "val/top1_collapse_rate" in val_metrics
    assert "val/routing_entropy" in val_metrics
    # Loss metrics still reported.
    assert "loss_total" in val_metrics
    assert "loss_action" in val_metrics


def test_validate_single_forward_pass() -> None:
    model = CountingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(num=16, seed=2), batch_size=4))

    trainer._validate()

    # 16 samples / batch 4 = 4 batches; the diagnostics must reuse the same
    # forward outputs instead of running a second pass.
    assert model.forward_calls == 4


def test_compute_loss_accepts_reused_output() -> None:
    model = CountingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(num=8, seed=3), batch_size=4))

    batch = DataLoader(_DictDataset(num=1, seed=4), batch_size=1)
    first = next(iter(batch))
    out = model(first)

    calls_before = model.forward_calls
    total_loss, metrics = trainer._compute_loss(first, out=out)
    assert model.forward_calls == calls_before
    assert total_loss.requires_grad
    assert set(metrics) == {"loss_total", "loss_action", "loss_balance"}


def test_expert_count_uses_configured_count_not_max_index() -> None:
    # Phases restricted to {0, 1} while the model has 3 experts: expert 2
    # never fires. The collapse rate must count it as dead (1/3) — deriving
    # the expert count from the largest observed index would report 0.5 and
    # hide the dead expert.
    model = CountingMoEModel(num_experts=3)
    dataset = _DictDataset(num=32, seed=6)
    dataset.phases.clamp_(max=1)
    trainer = _make_trainer(model, DataLoader(dataset, batch_size=8))

    val_metrics = trainer._validate()

    assert val_metrics["val/topk_collapse_rate"] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_fit_with_epoch_progressbar_completes() -> None:
    """The epoch-level tqdm progressbar path runs a full fit() cleanly.

    The progressbar wraps the epoch iterator, formats the postfix from
    validation metrics, and must be closed even on the normal path.
    """
    model = CountingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(num=16, seed=7), batch_size=8))
    trainer.train_cfg.epoch_progressbar = True

    trainer.fit()

    assert trainer.current_epoch == 1
    assert model.forward_calls > 0


def test_validate_losses_are_sample_weighted() -> None:
    # 7 samples in batches of 4 + 3: the short final batch must not weigh
    # as much as a full one. The reported loss is the mean over ALL samples.
    model = CountingMoEModel(num_experts=3)
    dataset = _DictDataset(num=7, seed=5)
    trainer = _make_trainer(model, DataLoader(dataset, batch_size=4))

    val_metrics = trainer._validate()

    # action_pred == 0 => per-sample MSE == action^2.
    expected = float((dataset.actions**2).mean())
    assert val_metrics["loss_action"] == pytest.approx(expected, rel=1e-6)
    assert val_metrics["loss_total"] == pytest.approx(expected, rel=1e-6)


def test_non_finite_loss_fails_fast() -> None:
    model = NaNMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(seed=8), batch_size=8))

    with pytest.raises(FloatingPointError, match="Non-finite training loss"):
        trainer.fit()


def test_non_finite_gradient_fails_fast() -> None:
    # The loss is finite (sqrt(0) = 0) but the gradient w.r.t. `a` is NaN
    # (0 * inf through the chain rule): only the gradient guard can catch it.
    model = SqrtPoisonModel()
    trainer = _make_trainer(model, DataLoader(_DictDataset(seed=9), batch_size=8))

    with pytest.raises(FloatingPointError, match="Non-finite gradient"):
        trainer.fit()


# ---------------------------------------------------------------------------
# Phase-utilization study: V4 (phase-conditional balance) and V2 (distillation)
# ---------------------------------------------------------------------------


def test_phase_conditional_balance_loss_added() -> None:
    # Each phase-p token routes one-hot to expert p: f_i^p = 1 for i = p,
    # p_i^p = 1, so the within-phase term is E * 1 per phase. Total loss =
    # coeff * E * (sum over phases of n_p/B) = coeff * E = 0.01 * 3.
    model = CountingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(num=32, seed=10), batch_size=8))
    trainer.train_cfg.phase_balance_coeff = 0.01

    batch = next(iter(DataLoader(_DictDataset(num=8, seed=11), batch_size=8)))
    total_loss, metrics = trainer._compute_loss(batch)

    assert "loss_phase_balance" in metrics
    assert metrics["loss_phase_balance"] == pytest.approx(0.01 * 3, abs=1e-4)
    assert "loss_balance" in metrics  # global balance still tracked


def test_phase_conditional_balance_off_by_default() -> None:
    model = CountingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(seed=12), batch_size=8))
    batch = next(iter(DataLoader(_DictDataset(num=8, seed=13), batch_size=8)))
    total_loss, metrics = trainer._compute_loss(batch)
    assert "loss_phase_balance" not in metrics


def test_phase_distill_warmup_adds_kl() -> None:
    # Teacher is one-hot phase 0 (softmax([10,0,0])); the router gate is
    # one-hot at the GT phase. For a phase-0 sample: KL ≈ log(1/softmax
    # (gate)_0)... with gate = [100, 0, 0], log q_0 ≈ 0, t_0 ≈ 1, so
    # KL ≈ 0. For phase-1 samples: log q_1 = log_softmax([100,0,0])_1 ≈
    # -100, t_0 ≈ 1, so KL ≈ t_0 * (log t_0 - log q_0) ≈ 100. With 1/3 of
    # samples at each phase the mean KL ≈ 2/3 * 100.
    model = PhaseEmittingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(num=32, seed=14), batch_size=8))
    trainer.train_cfg.phase_distill = {
        "enabled": True,
        "kappa0": 1.0,
        "tau": 1.0,
        "anneal_epochs": 40,
    }
    trainer.current_epoch = 0

    batch = next(iter(DataLoader(_DictDataset(num=12, seed=15), batch_size=12)))
    total_loss, metrics = trainer._compute_loss(batch)

    assert "loss_phase_distill" in metrics
    assert metrics["loss_phase_distill"] > 30.0  # dominated by the ~100/phase-1 KLs
    assert total_loss > metrics["loss_phase_distill"]


def test_phase_distill_anneals_to_zero() -> None:
    model = PhaseEmittingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(seed=16), batch_size=8))
    trainer.train_cfg.phase_distill = {
        "enabled": True,
        "kappa0": 1.0,
        "tau": 1.0,
        "anneal_epochs": 40,
    }
    trainer.current_epoch = 40

    batch = next(iter(DataLoader(_DictDataset(num=8, seed=17), batch_size=8)))
    total_loss, metrics = trainer._compute_loss(batch)
    assert "loss_phase_distill" not in metrics


def test_phase_distill_disabled_by_default() -> None:
    model = PhaseEmittingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(seed=18), batch_size=8))
    batch = next(iter(DataLoader(_DictDataset(num=8, seed=19), batch_size=8)))
    total_loss, metrics = trainer._compute_loss(batch)
    assert "loss_phase_distill" not in metrics
