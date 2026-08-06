"""CPU-only tests for model construction (registry + baseline stage tags)."""

from __future__ import annotations

from omegaconf import DictConfig

from phaseforge.models.baselines.oracle_moe import OraclePhaseMoEModel
from phaseforge.models.baselines.scratch_moe import ScratchMoEModel
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.expert import ExpertMLP
from phaseforge.models.components.router import TopKRouter
from phaseforge.utils.registry import build_model


def _make_baseline_cfg(target: str) -> DictConfig:
    return DictConfig(
        {
            "models": {
                "name": "test",
                "_target_": target,
                "freeze_encoder": False,
                "encoder": {
                    "_target_": "phaseforge.models.components.encoder.StateEncoder",
                    "input_dim": 23,
                    "hidden_dims": [64, 64],
                    "latent_dim": 32,
                },
                "router": {
                    "_target_": "phaseforge.models.components.router.TopKRouter",
                    "latent_dim": 32,
                    "num_experts": 4,
                    "top_k": 2,
                },
                "expert": {
                    "_target_": "phaseforge.models.components.expert.ExpertMLP",
                    "input_dim": 32,
                    "hidden_dims": [64],
                    "output_dim": 7,
                },
            }
        }
    )


def test_build_model_strips_freeze_encoder_and_name() -> None:
    cfg = _make_baseline_cfg("phaseforge.models.baselines.scratch_moe.ScratchMoEModel")
    model = build_model(cfg)
    assert isinstance(model, ScratchMoEModel)


def test_scratch_moe_reports_stage_2() -> None:
    encoder = StateEncoder(input_dim=23, hidden_dims=[64], latent_dim=32)
    router = TopKRouter(latent_dim=32, num_experts=4, top_k=2)
    expert = ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7)
    model = ScratchMoEModel(encoder=encoder, router=router, expert=expert)
    assert model.stage == 2


def test_oracle_moe_reports_stage_2() -> None:
    encoder = StateEncoder(input_dim=23, hidden_dims=[64], latent_dim=32)
    router = TopKRouter(latent_dim=32, num_experts=4, top_k=2)
    expert = ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7)
    model = OraclePhaseMoEModel(encoder=encoder, router=router, expert=expert, num_phases=3)
    assert model.stage == 2
