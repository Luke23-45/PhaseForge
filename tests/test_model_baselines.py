"""CPU-only tests for model construction (registry + baseline stage tags).

Also covers the 2x2 factorial cells (C1: ``phase_pretrain_random_router``,
``plain_encoder_phase_bootstrap``) and the teacher-forced cell (E8:
``teacher_forced``): construction via the registry, bootstrap semantics,
train/eval routing split, and Stage 1 checkpoint-source aliases.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from phaseforge.models.baselines.oracle_moe import OraclePhaseMoEModel
from phaseforge.models.baselines.phase_pretrain_random_router import (
    PhasePretrainRandomRouterModel,
)
from phaseforge.models.baselines.plain_encoder_phase_bootstrap import (
    PlainEncoderPhaseBootstrapModel,
)
from phaseforge.models.baselines.scratch_moe import ScratchMoEModel
from phaseforge.models.baselines.teacher_forced import TeacherForcedMoEModel
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.expert import ExpertMLP
from phaseforge.models.components.phase_head import PhaseClassificationHead
from phaseforge.models.components.router import TopKRouter
from phaseforge.utils.config import resolve_checkpoint_source
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
                    "input_dim": 151,
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


def _make_plain_cfg(target: str, num_phases: int = 3) -> DictConfig:
    cfg = _make_baseline_cfg(target)
    cfg.models.action_head = {
        "_target_": "phaseforge.models.components.action_head.ActionHead",
        "input_dim": 32,
        "output_dim": 7,
        "head_type": "deterministic",
        "hidden_dim": 64,
    }
    cfg.models.num_phases = num_phases
    return cfg


def _make_phase_cfg(target: str) -> DictConfig:
    cfg = _make_plain_cfg(target)
    cfg.models.phase_head = {
        "_target_": "phaseforge.models.components.phase_head.PhaseClassificationHead",
        "latent_dim": 32,
        "num_phases": 3,
    }
    # TeacherForcedMoEModel derives num_phases from the phase head.
    del cfg.models.num_phases
    return cfg


def _make_components(num_phases: int = 3):
    from phaseforge.models.components.action_head import ActionHead

    encoder = StateEncoder(input_dim=151, hidden_dims=[64], latent_dim=32)
    action_head = ActionHead(input_dim=32, output_dim=7, hidden_dim=64)
    router = TopKRouter(latent_dim=32, num_experts=4, top_k=2)
    expert = ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7)
    phase_head = PhaseClassificationHead(latent_dim=32, num_phases=num_phases)
    return encoder, action_head, router, expert, phase_head


class _DictDataset(Dataset):
    """Minimal dict-batch dataset matching the pipeline's batch keys."""

    def __init__(
        self,
        num: int = 64,
        input_dim: int = 151,
        num_phases: int = 3,
        seed: int = 0,
    ) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.states = torch.randn(num, input_dim, generator=gen)
        self.actions = torch.randn(num, 7, generator=gen)
        self.phases = torch.randint(0, num_phases, (num,), generator=gen)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "phase": self.phases[idx],
        }


def _make_dataloader(batch_size: int = 16) -> DataLoader:
    return DataLoader(_DictDataset(), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Registry construction
# ---------------------------------------------------------------------------


def test_build_model_strips_freeze_encoder_and_name() -> None:
    cfg = _make_baseline_cfg("phaseforge.models.baselines.scratch_moe.ScratchMoEModel")
    model = build_model(cfg)
    assert isinstance(model, ScratchMoEModel)


def test_scratch_moe_reports_stage_2() -> None:
    encoder = StateEncoder(input_dim=151, hidden_dims=[64], latent_dim=32)
    router = TopKRouter(latent_dim=32, num_experts=4, top_k=2)
    expert = ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7)
    model = ScratchMoEModel(encoder=encoder, router=router, expert=expert)
    assert model.stage == 2


def test_oracle_moe_reports_stage_2() -> None:
    encoder = StateEncoder(input_dim=151, hidden_dims=[64], latent_dim=32)
    router = TopKRouter(latent_dim=32, num_experts=4, top_k=2)
    expert = ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7)
    model = OraclePhaseMoEModel(encoder=encoder, router=router, expert=expert, num_phases=3)
    assert model.stage == 2


def test_phase_pretrain_random_router_builds() -> None:
    cfg = _make_plain_cfg(
        "phaseforge.models.baselines.phase_pretrain_random_router."
        "PhasePretrainRandomRouterModel"
    )
    # Warm-start structure takes no num_phases (no centroid bootstrap).
    del cfg.models.num_phases
    model = build_model(cfg)
    assert isinstance(model, PhasePretrainRandomRouterModel)


def test_plain_encoder_phase_bootstrap_builds() -> None:
    cfg = _make_plain_cfg(
        "phaseforge.models.baselines.plain_encoder_phase_bootstrap."
        "PlainEncoderPhaseBootstrapModel"
    )
    model = build_model(cfg)
    assert isinstance(model, PlainEncoderPhaseBootstrapModel)


def test_teacher_forced_builds() -> None:
    cfg = _make_phase_cfg(
        "phaseforge.models.baselines.teacher_forced.TeacherForcedMoEModel"
    )
    model = build_model(cfg)
    assert isinstance(model, TeacherForcedMoEModel)


# ---------------------------------------------------------------------------
# Stage 1 checkpoint-source aliases (C1 2x2 + E8 shared pretraining)
# ---------------------------------------------------------------------------


def test_resolve_checkpoint_source_new_cells() -> None:
    assert resolve_checkpoint_source("warmstart_moe") == "bc"
    assert resolve_checkpoint_source("plain_encoder_phase_bootstrap") == "bc"
    assert resolve_checkpoint_source("phase_pretrain_random_router") == "phaseforge"
    assert resolve_checkpoint_source("teacher_forced") == "phaseforge"
    assert resolve_checkpoint_source("bc") == "bc"
    assert resolve_checkpoint_source("phaseforge") == "phaseforge"


# ---------------------------------------------------------------------------
# phase_pretrain_random_router: random router must survive the bootstrap
# ---------------------------------------------------------------------------


def test_phase_pretrain_random_router_keeps_random_router() -> None:
    encoder, action_head, router, expert, _ = _make_components()
    model = PhasePretrainRandomRouterModel(
        encoder=encoder, action_head=action_head, router=router, expert=expert
    )
    assert model.stage == 1

    router_before = model.moe_layer.router.gate_linear.weight.data.clone()

    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")

    # Router must be untouched: the ONLY difference vs phaseforge is the
    # missing centroid init.
    assert model.stage == 2
    assert torch.allclose(model.moe_layer.router.gate_linear.weight.data, router_before)

    # Experts must be warm-started from the action head (mapping check).
    expert0 = model.moe_layer.experts[0]
    assert torch.allclose(expert0.hidden[0].weight, action_head.trunk[0].weight)
    assert torch.allclose(expert0.output_proj.weight, action_head.mean_head.weight)


# ---------------------------------------------------------------------------
# plain_encoder_phase_bootstrap: centroid bootstrap on a plain encoder
# ---------------------------------------------------------------------------


def test_plain_encoder_phase_bootstrap_sets_centroids() -> None:
    encoder, action_head, router, expert, _ = _make_components()
    model = PlainEncoderPhaseBootstrapModel(
        encoder=encoder, action_head=action_head, router=router, expert=expert,
        num_phases=3,
    )
    assert model.stage == 1

    # Expected centroids: mean latent per GT phase over the training data.
    dataloader = _make_dataloader()
    model.eval()
    phase_sums = torch.zeros(3, 32)
    phase_counts = torch.zeros(3)
    with torch.no_grad():
        for batch in dataloader:
            latent = model.encoder(batch["state"])
            phase = batch["phase"]
            phase_sums.scatter_add_(0, phase.unsqueeze(1).expand_as(latent), latent)
            phase_counts += torch.bincount(phase, minlength=3).float()
    expected = phase_sums / phase_counts.unsqueeze(1)
    expected = torch.nn.functional.normalize(expected, p=2, dim=-1)

    model.bootstrap_moe(dataloader=dataloader, device="cpu")

    assert model.stage == 2
    got = model.moe_layer.router.gate_linear.weight.data[:3]
    assert torch.allclose(got, expected, atol=1e-6)
    assert torch.allclose(
        model.moe_layer.router.gate_linear.bias.data, torch.zeros(4), atol=1e-6
    )

    # Experts warm-started from the action head.
    assert torch.allclose(
        model.moe_layer.experts[0].hidden[0].weight, action_head.trunk[0].weight
    )


# ---------------------------------------------------------------------------
# teacher_forced: GT-partitioned training, predicted-phase inference (E8)
# ---------------------------------------------------------------------------


def test_teacher_forced_training_routes_by_gt_phase() -> None:
    encoder, action_head, router, expert, phase_head = _make_components()
    model = TeacherForcedMoEModel(
        encoder=encoder, action_head=action_head, phase_head=phase_head,
        router=router, expert=expert,
    )
    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")
    assert model.stage == 2

    model.train()
    batch = _DictDataset(num=16)[0]
    out = model({k: v.unsqueeze(0) for k, v in batch.items()})
    assert out.expert_indices.squeeze(-1).item() == batch["phase"].item()
    assert torch.allclose(out.routing_weights, torch.ones(1, 1))
    assert out.aux_losses["balance"].item() == 0.0


def test_teacher_forced_eval_routes_by_predicted_phase() -> None:
    encoder, action_head, router, expert, phase_head = _make_components()
    model = TeacherForcedMoEModel(
        encoder=encoder, action_head=action_head, phase_head=phase_head,
        router=router, expert=expert,
    )
    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")

    model.eval()
    dataset = _DictDataset(num=16, seed=7)
    for i in range(16):
        batch = dataset[i]
        out = model({"state": batch["state"].unsqueeze(0)})
        with torch.no_grad():
            expected = phase_head(encoder(batch["state"].unsqueeze(0))).argmax(-1)
        # Routing follows the LEARNED phase predictor, never the GT label.
        assert out.expert_indices.squeeze(-1).item() == expected.item()


def test_teacher_forced_get_action_is_label_free() -> None:
    encoder, action_head, router, expert, phase_head = _make_components()
    model = TeacherForcedMoEModel(
        encoder=encoder, action_head=action_head, phase_head=phase_head,
        router=router, expert=expert,
    )
    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")
    model.eval()

    state = _DictDataset(num=1, seed=3)[0]["state"].unsqueeze(0)
    action = model.get_action(state)
    with torch.no_grad():
        latent = model.encoder(state)
        pred = phase_head(latent).argmax(-1)
        expected = model._dispatch(latent, pred).action_pred
    assert torch.allclose(action, expected, atol=1e-6)


def test_teacher_forced_freezes_stage1_bundle() -> None:
    encoder, action_head, router, expert, phase_head = _make_components()
    model = TeacherForcedMoEModel(
        encoder=encoder, action_head=action_head, phase_head=phase_head,
        router=router, expert=expert,
    )
    model.freeze_encoder()
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.phase_head.parameters())
    # Experts still trainable.
    assert any(p.requires_grad for p in model.moe_layer.experts[0].parameters())
