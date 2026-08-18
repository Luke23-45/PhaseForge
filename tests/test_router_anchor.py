"""CPU-only tests for the low-rank anchored router (V6) and the
phase-head router initialization (V1) in TopKRouter/PhaseBootstrappedMoE.

Covers: anchored construction (frozen backbone + zero-init low-rank
residual), the forward logit decomposition, validation errors, and the
bootstrap paths (router_init="phase_head" into the plain gate / into the
frozen anchor, plus the guarded combinations).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from phaseforge.models.components.router import TopKRouter
from phaseforge.models.phase_moe import PhaseBootstrappedMoE


class _DictDataset(Dataset):
    def __init__(self, num: int = 32, num_phases: int = 3, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.states = torch.randn(num, 8, generator=gen)
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


def _make_model(router: TopKRouter, num_phases: int = 3) -> PhaseBootstrappedMoE:
    from phaseforge.models.components.action_head import ActionHead
    from phaseforge.models.components.encoder import StateEncoder
    from phaseforge.models.components.expert import ExpertMLP
    from phaseforge.models.components.phase_head import PhaseClassificationHead

    encoder = StateEncoder(input_dim=8, hidden_dims=[16], latent_dim=8)
    action_head = ActionHead(input_dim=8, output_dim=2, hidden_dim=16)
    expert = ExpertMLP(input_dim=8, hidden_dims=[16], output_dim=2)
    phase_head = PhaseClassificationHead(latent_dim=8, num_phases=num_phases)
    return PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
    )


# ---------------------------------------------------------------------------
# Anchored router construction
# ---------------------------------------------------------------------------


def test_anchored_router_construction() -> None:
    router = TopKRouter(latent_dim=8, num_experts=3, anchor="phase_head", anchor_rank=2)
    assert isinstance(router.anchor_linear, nn.Linear)
    assert isinstance(router.gate_linear, nn.Sequential)
    assert len(router.gate_linear) == 2
    # The residual is zero-initialized: the router starts as the anchor.
    for lin in router.gate_linear:
        assert torch.allclose(lin.weight.data, torch.zeros_like(lin.weight.data))
        assert torch.allclose(lin.bias.data, torch.zeros_like(lin.bias.data))


def test_anchored_router_forward_decomposes() -> None:
    router = TopKRouter(latent_dim=8, num_experts=3, anchor="phase_head", anchor_rank=2)
    router.eval()  # suppress routing noise for the exact-logit check
    lat = torch.randn(5, 8)
    # Load a phase predictor into the anchor (as bootstrap_moe does): the
    # zero residual must leave the logits EXACTLY the anchor's.
    with torch.no_grad():
        router.anchor_linear.weight.copy_(torch.randn_like(router.anchor_linear.weight))
        router.anchor_linear.bias.copy_(torch.randn_like(router.anchor_linear.bias))
    with torch.inference_mode():
        expected = router.anchor_linear(lat)
        out = router(lat)
        assert out.gate_logits.shape == (5, 3)
        assert torch.allclose(out.gate_logits, expected, atol=1e-6)
        # Residual is trainable, anchor must be freezable.
        assert any(p.requires_grad for p in router.gate_linear.parameters())


def test_anchored_router_validation() -> None:
    with pytest.raises(ValueError, match="anchor must be None or 'phase_head'"):
        TopKRouter(latent_dim=8, num_experts=3, anchor="bogus")
    with pytest.raises(ValueError, match="anchor_rank must be a positive int"):
        TopKRouter(latent_dim=8, num_experts=3, anchor="phase_head", anchor_rank=0)
    with pytest.raises(ValueError, match="exceeds"):
        TopKRouter(latent_dim=8, num_experts=3, anchor="phase_head", anchor_rank=99)


def test_plain_router_has_no_anchor() -> None:
    router = TopKRouter(latent_dim=8, num_experts=3)
    assert router.anchor_linear is None
    assert isinstance(router.gate_linear, nn.Linear)


# ---------------------------------------------------------------------------
# bootstrap_moe: router_init="phase_head" (V1) and anchored (V6)
# ---------------------------------------------------------------------------


def test_bootstrap_phase_head_v1_copies_phase_head() -> None:
    router = TopKRouter(latent_dim=8, num_experts=3)
    model = _make_model(router)
    with torch.no_grad():
        model.phase_head.classifier.weight.copy_(torch.randn_like(model.phase_head.classifier.weight))
        model.phase_head.classifier.bias.copy_(torch.randn_like(model.phase_head.classifier.bias))
    w_before = model.phase_head.classifier.weight.data.clone()
    b_before = model.phase_head.classifier.bias.data.clone()

    model.bootstrap_moe(
        dataloader=DataLoader(_DictDataset(), batch_size=8),
        device="cpu",
        router_init="phase_head",
    )

    assert model.stage == 2
    assert torch.allclose(router.gate_linear.weight.data, w_before)
    assert torch.allclose(router.gate_linear.bias.data, b_before)
    # The phase head stays frozen (excluded from the Stage 2 graph).
    assert all(not p.requires_grad for p in model.phase_head.parameters())
    assert all(not p.requires_grad for p in model.action_head.parameters())


def test_bootstrap_anchored_v6_freezes_anchor() -> None:
    router = TopKRouter(latent_dim=8, num_experts=3, anchor="phase_head", anchor_rank=2)
    model = _make_model(router)
    with torch.no_grad():
        model.phase_head.classifier.weight.copy_(torch.randn_like(model.phase_head.classifier.weight))
    w_before = model.phase_head.classifier.weight.data.clone()

    model.bootstrap_moe(
        dataloader=DataLoader(_DictDataset(), batch_size=8),
        device="cpu",
        router_init="phase_head",
    )

    assert torch.allclose(router.anchor_linear.weight.data, w_before)
    assert all(not p.requires_grad for p in router.anchor_linear.parameters())
    # Residual still zero: initial routing is exactly the phase predictor.
    for lin in router.gate_linear:
        assert torch.allclose(lin.weight.data, torch.zeros_like(lin.weight.data))


def test_bootstrap_guard_combinations() -> None:
    model = _make_model(TopKRouter(latent_dim=8, num_experts=3))
    with pytest.raises(ValueError, match="router_init must be"):
        model.bootstrap_moe(DataLoader(_DictDataset()), device="cpu", router_init="bogus")

    anchored = _make_model(
        TopKRouter(latent_dim=8, num_experts=3, anchor="phase_head", anchor_rank=2)
    )
    with pytest.raises(ValueError, match="requires router_init='phase_head'"):
        anchored.bootstrap_moe(DataLoader(_DictDataset()), device="cpu", router_init="centroid")


def test_bootstrap_phase_head_requires_e_equals_p() -> None:
    router = TopKRouter(latent_dim=8, num_experts=4)
    model = _make_model(router, num_phases=3)
    with pytest.raises(ValueError, match="num_experts == num_phases"):
        model.bootstrap_moe(DataLoader(_DictDataset()), device="cpu", router_init="phase_head")