"""CPU-only tests for the Stage 2 trainer routing diagnostics (C3).

Verifies that every epoch's validation records the balance-vs-specialization
trajectory (``val/phase_expert_nmi``, ``val/balance_score``,
``val/collapse_rate``, ``val/routing_entropy``) and that validation reuses
the model forward outputs (no double forward pass).
"""

from __future__ import annotations

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
    return Stage2Trainer(
        cfg=cfg, model=model, train_loader=val_loader, val_loader=val_loader
    )


def test_validate_reports_routing_diagnostics() -> None:
    model = CountingMoEModel(num_experts=3)
    trainer = _make_trainer(model, DataLoader(_DictDataset(seed=1), batch_size=8))

    val_metrics = trainer._validate()

    # Perfect GT routing => perfect phase-expert alignment (NMI ~ 1.0) and
    # near-uniform usage => high balance score.
    assert "val/phase_expert_nmi" in val_metrics
    assert val_metrics["val/phase_expert_nmi"] > 0.99
    assert "val/balance_score" in val_metrics
    assert val_metrics["val/balance_score"] > 0.9
    assert "val/collapse_rate" in val_metrics
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
