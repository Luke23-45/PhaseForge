"""Contract tests for the matched temporal state-only BC baseline."""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize
from torch.utils.data import DataLoader

from phaseforge.data.common.collator import PhaseAwareCollator
from phaseforge.models.baselines.bc_rnn import BehaviorCloningRNNModel
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.utils.registry import build_model, build_trainer


def _model(rnn_type: str = "lstm") -> BehaviorCloningRNNModel:
    return BehaviorCloningRNNModel(
        encoder=StateEncoder(
            input_dim=5,
            hidden_dims=[8],
            latent_dim=6,
            dropout=0.0,
        ),
        action_head=ActionHead(input_dim=7, output_dim=3, hidden_dim=8),
        hidden_dim=7,
        num_layers=2,
        rnn_type=rnn_type,
    )


def test_forward_supports_single_step_and_windows() -> None:
    model = _model()
    single = model({"state": torch.randn(4, 5)})
    sequence = model({"state": torch.randn(4, 6, 5)})
    assert single.action_pred.shape == (4, 3)
    assert sequence.action_pred.shape == (4, 6, 3)


def test_forward_gru_variant() -> None:
    model = _model(rnn_type="gru")
    single = model({"state": torch.randn(4, 5)})
    sequence = model({"state": torch.randn(4, 6, 5)})
    assert single.action_pred.shape == (4, 3)
    assert sequence.action_pred.shape == (4, 6, 3)


def test_streaming_state_is_reset_between_episodes() -> None:
    model = _model().eval()
    first = model.get_action(torch.zeros(1, 5))
    _ = model.get_action(torch.ones(1, 5))
    model.reset()
    replay = model.get_action(torch.zeros(1, 5))
    assert torch.allclose(first, replay)


def test_complete_sequence_does_not_use_streaming_history() -> None:
    model = _model().eval()
    _ = model.get_action(torch.ones(1, 5))
    seq = torch.randn(1, 4, 5)
    from_sequence = model.get_action(seq)
    model.reset()
    expected = model.get_action(seq)
    assert torch.allclose(from_sequence, expected)


def test_invalid_parameters_rejected() -> None:
    encoder = StateEncoder(input_dim=5, hidden_dims=[8], latent_dim=6)
    action_head = ActionHead(input_dim=7, output_dim=3, hidden_dim=8)
    with pytest.raises(ValueError, match="hidden_dim"):
        BehaviorCloningRNNModel(encoder=encoder, action_head=action_head, hidden_dim=0)
    with pytest.raises(ValueError, match="num_layers"):
        BehaviorCloningRNNModel(encoder=encoder, action_head=action_head, num_layers=0)
    with pytest.raises(ValueError, match="rnn_type"):
        BehaviorCloningRNNModel(encoder=encoder, action_head=action_head, rnn_type="transformer")
    with pytest.raises(ValueError, match="dropout"):
        BehaviorCloningRNNModel(encoder=encoder, action_head=action_head, dropout=1.5)


def test_build_model_from_hydra_config() -> None:
    with initialize(version_base="1.3", config_path="../phaseforge/config"):
        cfg = compose(config_name="main", overrides=["models=baselines/bc_rnn", "data=lift_rnn"])
        model = build_model(cfg)
        assert isinstance(model, BehaviorCloningRNNModel)
        assert model.num_parameters() > 0


def test_stage1_trainer_with_padded_sequence_batch() -> None:
    with initialize(version_base="1.3", config_path="../phaseforge/config"):
        cfg = compose(
            config_name="main",
            overrides=[
                "models=baselines/bc_rnn",
                "data=lift_rnn",
                "train=stage1",
                "project.device=cpu",
            ],
        )
        model = build_model(cfg)
        batch = [
            {
                "state": torch.randn(10, 19),
                "action": torch.randn(10, 7),
                "phase": torch.randint(0, 6, (10,)),
                "task_id": 0,
                "trajectory_id": 0,
                "trajectory_position": 0,
            },
            {
                "state": torch.randn(8, 19),
                "action": torch.randn(8, 7),
                "phase": torch.randint(0, 6, (8,)),
                "task_id": 0,
                "trajectory_id": 1,
                "trajectory_position": 0,
            },
        ]
        collator = PhaseAwareCollator()
        loader = DataLoader(batch, batch_size=2, collate_fn=collator)
        trainer = build_trainer(cfg, model, loader, loader)
        trainer._train_epoch()
        val_metrics = trainer._validate()
        assert "loss_total" in val_metrics
        assert "loss_action" in val_metrics
