"""CPU-only tests for the offline evaluator and its decisive action-MSE metric.

``eval/action_mse`` is the fast diagnostic that distinguishes "the model
cannot reproduce the demonstration actions" (0% LIBERO rollout success is
expected) from "the eval path is broken" — and it is logged immediately so
an interrupted run (e.g. inside the slower routing metrics) still shows the
decisive number.
"""

from __future__ import annotations

import logging

import pytest
import torch
from omegaconf import OmegaConf

from phaseforge.evaluations.metrics.task_metrics import action_mse
from phaseforge.evaluations.runners.offline_evaluator import OfflineEvaluator
from phaseforge.models.base import BaseManipulationModel, ModelOutput


class FakeModel(BaseManipulationModel):
    """Deterministic policy: action = first 7 state dims doubled."""

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        return ModelOutput(action_pred=batch["state"][:, :7] * 2.0)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, :7] * 2.0

    def num_parameters(self) -> int:
        return 0


def _make_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "project": {"device": "cpu"},
            "eval": {
                "task": {
                    "success_rate": {"enabled": True, "l2_threshold": 0.05},
                    "boundary_smoothness": {"enabled": False},
                },
            },
        }
    )


# ---------------------------------------------------------------------------
# action_mse
# ---------------------------------------------------------------------------


def test_action_mse_exact_match() -> None:
    preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert action_mse(preds, preds) == 0.0


def test_action_mse_known_value() -> None:
    preds = torch.zeros(2, 2)
    targets = torch.tensor([[1.0, 3.0], [1.0, 3.0]])
    # (1^2 + 3^2 + 1^2 + 3^2) / 4 = 20 / 4 = 5
    assert action_mse(preds, targets) == pytest.approx(5.0)


def test_action_mse_respects_padding_mask() -> None:
    preds = torch.zeros(2, 3, 2)  # (B, T, A)
    targets = torch.ones(2, 3, 2)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])  # (B, T): 3 valid entries
    assert action_mse(preds, targets, mask) == pytest.approx(1.0)


def test_action_mse_empty_returns_nan() -> None:
    preds = torch.zeros(0, 2)
    targets = torch.zeros(0, 2)
    assert torch.isnan(torch.tensor(action_mse(preds, targets)))


# ---------------------------------------------------------------------------
# OfflineEvaluator.run()
# ---------------------------------------------------------------------------


def test_run_reports_and_logs_action_mse_immediately(caplog) -> None:
    cfg = _make_cfg()
    states = torch.randn(8, 23)
    actions = torch.randn(8, 7)
    dataloader = [
        {"state": states[:4], "action": actions[:4], "phase": torch.zeros(4, dtype=torch.long)},
        {"state": states[4:], "action": actions[4:], "phase": torch.zeros(4, dtype=torch.long)},
    ]

    evaluator = OfflineEvaluator(cfg=cfg, model=FakeModel(), dataloader=dataloader)
    expected_mse = float(((states[:, :7] * 2.0 - actions) ** 2).mean())

    with caplog.at_level(logging.INFO, logger="phaseforge.evaluations.runners.offline_evaluator"):
        results = evaluator.run()

    assert results["eval/action_mse"] == pytest.approx(expected_mse)
    assert "eval/action_mse" in caplog.text
    assert "eval/success_rate" in caplog.text
    assert "eval/success_rate" in results
    assert results["eval/action_mse"] > 0.0  # the fake model is wrong on purpose
