"""CPU-only tests for model construction (registry + baseline stage tags).

Also covers the 2x2 factorial cells (C1: ``phase_pretrain_random_router``,
``plain_encoder_phase_bootstrap``) and the teacher-forced cell (E8:
``teacher_forced``): construction via the registry, bootstrap semantics,
train/eval routing split, and Stage 1 checkpoint-source aliases.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
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
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.expert import (
    ExpertMLP,
    warm_start_experts_from_action_head,
)
from phaseforge.models.components.moe_layer import MoELayer
from phaseforge.models.components.phase_head import PhaseClassificationHead
from phaseforge.models.components.router import TopKRouter
from phaseforge.models.phase_moe import PhaseBootstrappedMoE
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
        "phaseforge.models.baselines.phase_pretrain_random_router.PhasePretrainRandomRouterModel"
    )
    # Warm-start structure takes no num_phases (no centroid bootstrap).
    del cfg.models.num_phases
    model = build_model(cfg)
    assert isinstance(model, PhasePretrainRandomRouterModel)


def test_plain_encoder_phase_bootstrap_builds() -> None:
    cfg = _make_plain_cfg(
        "phaseforge.models.baselines.plain_encoder_phase_bootstrap.PlainEncoderPhaseBootstrapModel"
    )
    model = build_model(cfg)
    assert isinstance(model, PlainEncoderPhaseBootstrapModel)


def test_teacher_forced_builds() -> None:
    cfg = _make_phase_cfg("phaseforge.models.baselines.teacher_forced.TeacherForcedMoEModel")
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

    torch.manual_seed(1234)
    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")

    # Router must be untouched: the ONLY difference vs phaseforge is the
    # missing centroid init.
    assert model.stage == 2
    assert torch.allclose(model.moe_layer.router.gate_linear.weight.data, router_before)

    # Experts must be warm-started from the action head (mapping check).
    # The bootstrap adds a small symmetry-breaking jitter (std 0.02), so the
    # copy matches within tolerance rather than bit-for-bit.
    expert0 = model.moe_layer.experts[0]
    assert torch.allclose(expert0.hidden[0].weight, action_head.trunk[0].weight, atol=0.1)
    assert torch.allclose(expert0.output_proj.weight, action_head.mean_head.weight, atol=0.1)
    # ...but the jitter must break the bit-identical symmetry between experts.
    assert not torch.allclose(
        model.moe_layer.experts[0].output_proj.weight,
        model.moe_layer.experts[1].output_proj.weight,
    )


# ---------------------------------------------------------------------------
# plain_encoder_phase_bootstrap: centroid bootstrap on a plain encoder
# ---------------------------------------------------------------------------


def test_plain_encoder_phase_bootstrap_sets_centroids() -> None:
    encoder, action_head, router, expert, _ = _make_components()
    model = PlainEncoderPhaseBootstrapModel(
        encoder=encoder,
        action_head=action_head,
        router=router,
        expert=expert,
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

    torch.manual_seed(1234)
    model.bootstrap_moe(dataloader=dataloader, device="cpu")

    assert model.stage == 2
    got = model.moe_layer.router.gate_linear.weight.data[:3]
    assert torch.allclose(got, expected, atol=1e-6)
    assert torch.allclose(model.moe_layer.router.gate_linear.bias.data, torch.zeros(4), atol=1e-6)

    # Experts warm-started from the action head (within the symmetry-breaking
    # jitter tolerance; same reasoning as the random-router test above).
    assert torch.allclose(
        model.moe_layer.experts[0].hidden[0].weight,
        action_head.trunk[0].weight,
        atol=0.1,
    )
    assert torch.allclose(
        model.moe_layer.experts[0].output_proj.weight,
        action_head.mean_head.weight,
        atol=0.1,
    )


# ---------------------------------------------------------------------------
# teacher_forced: GT-partitioned training, predicted-phase inference (E8)
# ---------------------------------------------------------------------------


def test_teacher_forced_training_routes_by_gt_phase() -> None:
    encoder, action_head, router, expert, phase_head = _make_components()
    model = TeacherForcedMoEModel(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
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
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
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
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
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
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
    )
    model.freeze_encoder()
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.phase_head.parameters())
    # Experts still trainable.
    assert any(p.requires_grad for p in model.moe_layer.experts[0].parameters())


# ---------------------------------------------------------------------------
# Warm start & bootstrap hardening
# ---------------------------------------------------------------------------


def test_warm_start_exact_copy_when_jitter_disabled() -> None:
    _, action_head, _, _, _ = _make_components()
    expert = ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7)
    warm_start_experts_from_action_head(nn.ModuleList([expert]), action_head, jitter_std=0.0)
    assert torch.equal(expert.hidden[0].weight, action_head.trunk[0].weight)
    assert torch.equal(expert.output_proj.weight, action_head.mean_head.weight)


def test_warm_start_rejects_architecture_mismatch() -> None:
    _, action_head, _, _, _ = _make_components()
    # A two-hidden-layer expert cannot be filled by the single-hidden-layer
    # ActionHead: the warm start must fail loudly, not copy partially.
    deep_expert = ExpertMLP(input_dim=32, hidden_dims=[64, 64], output_dim=7)
    with pytest.raises(ValueError, match="cannot fill"):
        warm_start_experts_from_action_head(nn.ModuleList([deep_expert]), action_head)


def test_warm_start_rejects_negative_jitter() -> None:
    _, action_head, _, _, _ = _make_components()
    expert = ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7)
    with pytest.raises(ValueError, match="jitter_std"):
        warm_start_experts_from_action_head(nn.ModuleList([expert]), action_head, jitter_std=-0.1)


def test_warm_started_experts_are_independent_copies() -> None:
    encoder, action_head, router, expert, _ = _make_components()
    model = PhasePretrainRandomRouterModel(
        encoder=encoder, action_head=action_head, router=router, expert=expert
    )
    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")

    before = model.moe_layer.experts[1].output_proj.weight.detach().clone()
    with torch.no_grad():
        model.moe_layer.experts[0].output_proj.weight.mul_(0.0)
    # Clones share no storage: perturbing expert 0 must leave expert 1 intact.
    assert torch.allclose(model.moe_layer.experts[1].output_proj.weight, before)


def test_moe_layer_clones_have_independent_storage() -> None:
    router = TopKRouter(latent_dim=32, num_experts=4, top_k=2)
    expert = ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7)
    layer = MoELayer(router=router, experts=expert)
    assert len(layer.experts) == 4
    template_ptr = expert.hidden[0].weight.data_ptr()
    for clone in layer.experts:
        assert clone is not expert
        # Re-initialized deepcopy: separate storage from the template.
        assert clone.hidden[0].weight.data_ptr() != template_ptr


def test_plain_bootstrap_rejects_absent_phase() -> None:
    encoder, action_head, router, expert, _ = _make_components()
    model = PlainEncoderPhaseBootstrapModel(
        encoder=encoder,
        action_head=action_head,
        router=router,
        expert=expert,
        num_phases=3,
    )
    # Restrict the phases to {0, 1}: phase 2 never appears.
    dataset = _DictDataset(num=64, seed=1)
    dataset.phases.clamp_(max=1)
    dataloader = DataLoader(dataset, batch_size=16)
    with pytest.raises(ValueError, match="zero samples"):
        model.bootstrap_moe(dataloader=dataloader, device="cpu")
    # The bootstrap aborted before the stage transition.
    assert model.stage == 1


def test_bootstrap_freezes_unused_stage1_heads() -> None:
    encoder, action_head, router, expert, phase_head = _make_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
    )
    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")
    assert model.stage == 2
    assert all(not p.requires_grad for p in model.action_head.parameters())
    assert all(not p.requires_grad for p in model.phase_head.parameters())
    # Experts remain trainable for Stage 2.
    assert any(p.requires_grad for p in model.moe_layer.experts[0].parameters())


def test_frozen_encoder_stays_in_eval_after_train() -> None:
    encoder, action_head, router, expert, phase_head = _make_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
    )
    model.freeze_encoder()
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert not encoder.training
    model.train()
    # The override keeps the frozen encoder deterministic (dropout off) even
    # though the rest of the model is in training mode.
    assert model.training
    assert not encoder.training


def test_oracle_requires_phase_labels() -> None:
    encoder, _, router, expert, _ = _make_components()
    model = OraclePhaseMoEModel(encoder=encoder, router=router, expert=expert, num_phases=3)
    with pytest.raises(RuntimeError, match="requires ground-truth"):
        model({"state": torch.randn(4, 151)})


def test_router_rejects_invalid_configs() -> None:
    with pytest.raises(ValueError, match="positive int"):
        TopKRouter(latent_dim=32, num_experts=0, top_k=1)
    with pytest.raises(ValueError, match="positive int"):
        TopKRouter(latent_dim=32, num_experts=4, top_k=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        TopKRouter(latent_dim=32, num_experts=4, top_k=5)
    with pytest.raises(ValueError, match="noise_std"):
        TopKRouter(latent_dim=32, num_experts=4, noise_std=-0.1)
    with pytest.raises(ValueError, match="balance_coeff"):
        TopKRouter(latent_dim=32, num_experts=4, balance_coeff=-1.0)


def test_router_normalize_input_produces_cosine_logits() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 32)
    centroids = torch.nn.functional.normalize(torch.randn(4, 32), dim=-1)
    router = TopKRouter(
        latent_dim=32,
        num_experts=4,
        top_k=2,
        noise_std=0.0,
        normalize_input=True,
    )
    router.gate_linear.weight.data.copy_(centroids)
    router.gate_linear.bias.data.zero_()
    router.eval()

    out = router(x)
    # Unit-norm latents x unit-norm centroid weights => true cosine logits.
    expected = torch.nn.functional.normalize(x, dim=-1) @ centroids.T
    assert torch.allclose(out.gate_logits, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Action contract: tanh-squashed outputs must stay in (-1, 1)
# ---------------------------------------------------------------------------


def test_action_head_output_is_bounded_to_unit_interval() -> None:
    """The final tanh must bound ActionHead predictions to (-1, 1) even for
    large raw activations (the robomimic-standard action contract)."""
    head = ActionHead(input_dim=32, output_dim=7, hidden_dim=64)
    # Large magnitude latent forces raw mean_head outputs well outside
    # [-1, 1]; without the tanh this test would fail.
    latent = torch.full((8, 32), 5.0)
    action = head(latent)
    assert action.shape == (8, 7)
    assert torch.isfinite(action).all()
    assert float(action.detach().min()) > -1.0
    assert float(action.detach().max()) < 1.0


def test_expert_output_is_bounded_to_unit_interval() -> None:
    """MoE experts must also honour the [-1, 1] action contract."""
    expert = ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7)
    latent = torch.full((8, 32), 5.0)
    action = expert(latent)
    assert action.shape == (8, 7)
    assert torch.isfinite(action).all()
    assert float(action.detach().min()) > -1.0
    assert float(action.detach().max()) < 1.0


def test_get_action_is_bounded_in_both_stages() -> None:
    """get_action() must produce in-contract actions in Stage 1 (ActionHead)
    and Stage 2 (MoE experts) so rollout evaluation never trips the strict
    validate_action contract."""
    encoder, action_head, router, expert, phase_head = _make_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
    )
    model.eval()
    state = torch.full((1, 151), 5.0)

    with torch.inference_mode():
        stage1_action = model.get_action(state)
        assert float(stage1_action.min()) > -1.0
        assert float(stage1_action.max()) < 1.0

        model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")
        assert model.stage == 2
        stage2_action = model.get_action(state)
        assert float(stage2_action.min()) > -1.0
        assert float(stage2_action.max()) < 1.0
