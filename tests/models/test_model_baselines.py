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
from phaseforge.models.baselines.warmstart_moe import WarmStartMoEModel
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.expert import (
    ExpertMLP,
    one_warm_experts_from_action_head,
    partial_reinit_experts_from_action_head,
    warm_start_experts_from_action_head,
)
from phaseforge.models.components.moe_layer import MoELayer
from phaseforge.models.components.phase_head import PhaseClassificationHead
from phaseforge.models.components.router import TopKRouter
from phaseforge.models.components.soft_mapping import build_hierarchical_uniform_mapping
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
    """R50-matched C1 cell: random router, 50% partial warm-start experts.

    The control's registered configuration pins ``expert_init`` to the
    canonical partial warm-start, so the bootstrap must (a) leave the router
    untouched and (b) initialize experts exactly like the proposed method:
    kept neurons bit-exact copies of the ActionHead, dropped neurons
    reinitialized per expert.
    """
    encoder, action_head, router, expert, _ = _make_components()
    model = PhasePretrainRandomRouterModel(
        encoder=encoder,
        action_head=action_head,
        router=router,
        expert=expert,
        expert_init={"type": "partial_warm", "drop_rate": 0.5, "seed": 42},
    )
    assert model.stage == 1

    router_before = model.moe_layer.router.gate_linear.weight.data.clone()

    torch.manual_seed(1234)
    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu", training_seed=42)

    # Router must be untouched: the ONLY difference vs phaseforge is the
    # missing centroid init.
    assert model.stage == 2
    assert torch.allclose(model.moe_layer.router.gate_linear.weight.data, router_before)

    # Experts: partial warm-start semantics (hidden dim 64 -> 32 dropped).
    info = model._expert_init_info
    assert info["expert_init"]["type"] == "partial_warm"
    assert info["expert_init"]["drop_rate"] == 0.5
    assert info["expert_init"]["init_seed"] == 42
    assert info["expert_init"]["num_dropped_neurons"] == 32
    dropped = info["expert_init"]["dropped_neuron_indices"]
    assert len(dropped) == 32 and len(set(dropped)) == 32
    assert info["expert_init"]["dropped_indices_sha256"]
    assert info["router"]["init_type"] == "random"
    assert info["training_seed"] == 42

    kept = [i for i in range(64) if i not in dropped]
    expert0 = model.moe_layer.experts[0]
    # Kept rows: bit-exact ActionHead copies (partial warm = no jitter).
    assert torch.equal(expert0.hidden[0].weight.data[kept], action_head.trunk[0].weight.data[kept])
    # Dropped rows: reinitialized (differ from the ActionHead copy).
    assert not torch.equal(
        expert0.hidden[0].weight.data[dropped], action_head.trunk[0].weight.data[dropped]
    )
    # Dropped-row biases zeroed and experts differ from each other on drops.
    assert torch.all(expert0.hidden[0].bias.data[dropped] == 0.0)
    assert not torch.equal(
        model.moe_layer.experts[0].hidden[0].weight.data,
        model.moe_layer.experts[1].hidden[0].weight.data,
    )


# ---------------------------------------------------------------------------
# plain_encoder_phase_bootstrap: centroid bootstrap on a plain encoder
# ---------------------------------------------------------------------------


def test_plain_encoder_phase_bootstrap_sets_centroids() -> None:
    """R50-matched C1 cell: BC-encoder centroids + 50% partial warm experts."""
    encoder, action_head, router, expert, _ = _make_components()
    model = PlainEncoderPhaseBootstrapModel(
        encoder=encoder,
        action_head=action_head,
        router=router,
        expert=expert,
        num_phases=3,
        expert_init={"type": "partial_warm", "drop_rate": 0.5, "seed": 42},
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
    model.bootstrap_moe(dataloader=dataloader, device="cpu", training_seed=42)

    assert model.stage == 2
    got = model.moe_layer.router.gate_linear.weight.data[:3]
    assert torch.allclose(got, expected, atol=1e-6)
    assert torch.allclose(model.moe_layer.router.gate_linear.bias.data, torch.zeros(4), atol=1e-6)

    # Experts: partial warm-start semantics (hidden dim 64 -> 32 dropped).
    info = model._expert_init_info
    assert info["expert_init"]["type"] == "partial_warm"
    assert info["expert_init"]["drop_rate"] == 0.5
    assert info["expert_init"]["num_dropped_neurons"] == 32
    assert info["expert_init"]["dropped_indices_sha256"]
    assert info["router"]["init_type"] == "centroid"
    assert info["training_seed"] == 42

    dropped = info["expert_init"]["dropped_neuron_indices"]
    kept = [i for i in range(64) if i not in dropped]
    expert0 = model.moe_layer.experts[0]
    assert torch.equal(expert0.hidden[0].weight.data[kept], action_head.trunk[0].weight.data[kept])
    assert not torch.equal(
        expert0.hidden[0].weight.data[dropped], action_head.trunk[0].weight.data[dropped]
    )


# ---------------------------------------------------------------------------
# R50-matched controls: CLI parity, seed determinism, config contract
# ---------------------------------------------------------------------------


def test_bootstrap_accepts_cli_training_seed_kwarg() -> None:
    """Every bootstrap class must accept the CLI's exact bootstrap call.

    Regression guard: ``phaseforge/cli.py`` passes ``training_seed=`` to
    ``bootstrap_moe`` unconditionally. Commit 0a7e415 added the kwarg to the
    CLI without updating the baseline signatures, so every baseline stage-2
    run crashed with a TypeError until the signatures were aligned.
    """
    cases = [
        lambda: WarmStartMoEModel(
            encoder=StateEncoder(input_dim=151, hidden_dims=[64], latent_dim=32),
            action_head=ActionHead(input_dim=32, output_dim=7, hidden_dim=64),
            router=TopKRouter(latent_dim=32, num_experts=4, top_k=2),
            expert=ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7),
        ),
        lambda: PhasePretrainRandomRouterModel(
            encoder=StateEncoder(input_dim=151, hidden_dims=[64], latent_dim=32),
            action_head=ActionHead(input_dim=32, output_dim=7, hidden_dim=64),
            router=TopKRouter(latent_dim=32, num_experts=4, top_k=2),
            expert=ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7),
            expert_init={"type": "partial_warm", "drop_rate": 0.5, "seed": 42},
        ),
        lambda: PlainEncoderPhaseBootstrapModel(
            encoder=StateEncoder(input_dim=151, hidden_dims=[64], latent_dim=32),
            action_head=ActionHead(input_dim=32, output_dim=7, hidden_dim=64),
            router=TopKRouter(latent_dim=32, num_experts=4, top_k=2),
            expert=ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7),
            num_phases=3,
            expert_init={"type": "partial_warm", "drop_rate": 0.5, "seed": 42},
        ),
        lambda: TeacherForcedMoEModel(
            encoder=StateEncoder(input_dim=151, hidden_dims=[64], latent_dim=32),
            action_head=ActionHead(input_dim=32, output_dim=7, hidden_dim=64),
            phase_head=PhaseClassificationHead(latent_dim=32, num_phases=3),
            router=TopKRouter(latent_dim=32, num_experts=4, top_k=2),
            expert=ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7),
        ),
    ]
    for make in cases:
        model = make()
        model.bootstrap_moe(
            dataloader=_make_dataloader(), device="cpu", training_seed=42
        )
        assert model.stage == 2
        assert model._expert_init_info is not None
        assert model._expert_init_info["training_seed"] == 42


def test_warmstart_moe_default_remains_standard_warmstart() -> None:
    """S3.10 guard: the behavioral baseline must keep its full warm start.

    ``warmstart_moe`` is not an R50 factorial control; converting it would
    silently change the registered behavioral baseline.
    """
    encoder, action_head, router, expert, _ = _make_components()
    model = WarmStartMoEModel(
        encoder=encoder, action_head=action_head, router=router, expert=expert
    )
    torch.manual_seed(1234)
    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu", training_seed=42)

    info = model._expert_init_info
    assert info["expert_init"]["type"] == "warmstart"
    assert info["expert_init"]["jitter_std"] == 0.02
    assert "drop_rate" not in info["expert_init"]
    # Full copy (with jitter tolerance, NOT bit-exact: jitter breaks symmetry).
    expert0 = model.moe_layer.experts[0]
    assert torch.allclose(expert0.hidden[0].weight, action_head.trunk[0].weight, atol=0.1)
    assert not torch.equal(expert0.hidden[0].weight, action_head.trunk[0].weight)


def test_group_b_partial_warm_is_seed_deterministic() -> None:
    """Same init seed -> bit-identical experts; different seed -> different.

    All builds start from deep-copied (bit-identical) components so the only
    varying factor is the partial-warm init seed.
    """
    import copy

    encoder, action_head, router, expert, _ = _make_components()

    def build_and_bootstrap(seed: int) -> PhasePretrainRandomRouterModel:
        model = PhasePretrainRandomRouterModel(
            encoder=copy.deepcopy(encoder),
            action_head=copy.deepcopy(action_head),
            router=copy.deepcopy(router),
            expert=copy.deepcopy(expert),
            expert_init={"type": "partial_warm", "drop_rate": 0.5, "seed": seed},
        )
        model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")
        return model

    a1 = build_and_bootstrap(42)
    a2 = build_and_bootstrap(42)
    b = build_and_bootstrap(43)

    for e_a, e_b in zip(a1.moe_layer.experts, a2.moe_layer.experts):
        assert torch.equal(e_a.hidden[0].weight, e_b.hidden[0].weight)
    for e_a, e_b in zip(a1.moe_layer.experts, b.moe_layer.experts):
        assert not torch.equal(e_a.hidden[0].weight, e_b.hidden[0].weight)
    assert (
        a1._expert_init_info["expert_init"]["dropped_indices_sha256"]
        == a2._expert_init_info["expert_init"]["dropped_indices_sha256"]
    )
    assert (
        a1._expert_init_info["expert_init"]["dropped_indices_sha256"]
        != b._expert_init_info["expert_init"]["dropped_indices_sha256"]
    )


def test_group_b_rejects_unknown_expert_init_type() -> None:
    for make_model in (
        lambda: WarmStartMoEModel(
            encoder=StateEncoder(input_dim=151, hidden_dims=[64], latent_dim=32),
            action_head=ActionHead(input_dim=32, output_dim=7, hidden_dim=64),
            router=TopKRouter(latent_dim=32, num_experts=4, top_k=2),
            expert=ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7),
            expert_init={"type": "not_a_real_type"},
        ),
        lambda: PlainEncoderPhaseBootstrapModel(
            encoder=StateEncoder(input_dim=151, hidden_dims=[64], latent_dim=32),
            action_head=ActionHead(input_dim=32, output_dim=7, hidden_dim=64),
            router=TopKRouter(latent_dim=32, num_experts=4, top_k=2),
            expert=ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7),
            num_phases=3,
            expert_init={"type": "not_a_real_type"},
        ),
    ):
        model = make_model()
        with pytest.raises(ValueError, match="Unknown expert_init type"):
            model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")
        assert model.stage == 1


def test_r50_matched_control_configs_resolve_partial_warm() -> None:
    """S3.9 config guard: the five matched controls pin the R50 expert init.

    Composes every control's registered model config and asserts the partial
    warm-start contract, including that the init seed follows the training
    seed (not a constant). The behavioral baselines (warmstart_moe,
    scratch_moe) and the teacher-forced diagnostic must NOT carry an
    expert_init override (S3.10).
    """
    from hydra import compose, initialize

    matched = [
        "baselines/phase_pretrain_random_router",
        "baselines/plain_encoder_phase_bootstrap",
        "baselines/pf_spherical_kmeans",
        "baselines/pf_kmeans",
        "baselines/pf_phase_head",
    ]
    untouched = ["baselines/warmstart_moe", "baselines/scratch_moe", "baselines/teacher_forced"]

    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        for seed in (42, 43):
            for model_path in matched:
                cfg = compose(
                    config_name="main",
                    overrides=[
                        f"models={model_path}",
                        "train=stage2",
                        "data=common",
                        f"project.seed={seed}",
                    ],
                )
                ei = cfg.models.expert_init
                assert ei.type == "partial_warm", f"{model_path} must pin partial_warm"
                assert ei.drop_rate == 0.5, f"{model_path} must pin drop_rate=0.5"
                assert ei.jitter_std == 0.0, f"{model_path} must pin jitter_std=0.0"
                assert ei.seed == seed, (
                    f"{model_path} init seed must follow the training seed"
                )
        for model_path in untouched:
            cfg = compose(
                config_name="main",
                overrides=[f"models={model_path}", "train=stage2", "data=common"],
            )
            assert "expert_init" not in cfg.models, (
                f"{model_path} must NOT carry an expert_init override"
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


def test_partial_reinit_keeps_and_reinit_split() -> None:
    _, action_head, _, _, _ = _make_components()
    hidden_dim = 64
    experts = nn.ModuleList(
        [ExpertMLP(input_dim=32, hidden_dims=[hidden_dim], output_dim=7) for _ in range(2)]
    )

    torch.manual_seed(0)
    action_head.trunk[0].weight.data.normal_()
    action_head.trunk[0].bias.data.normal_()
    action_head.mean_head.weight.data.normal_()
    action_head.mean_head.bias.data.normal_()

    drop_rate = 0.5
    partial_reinit_experts_from_action_head(
        experts, action_head, drop_rate=drop_rate, seed=42
    )

    k = int(round(drop_rate * hidden_dim))
    expected_dropped = sorted(
        torch.randperm(
            hidden_dim, generator=torch.Generator().manual_seed(42)
        )[:k].tolist()
    )
    expected_kept = sorted(set(range(hidden_dim)) - set(expected_dropped))

    for expert in experts:
        for i in expected_dropped:
            assert not torch.equal(
                expert.hidden[0].weight.data[i, :],
                action_head.trunk[0].weight.data[i, :],
            ), f"dropped neuron {i} must be reinitialized (not equal to ActionHead)"
            assert expert.hidden[0].bias.data[i].item() == 0.0
            assert not torch.equal(
                expert.output_proj.weight.data[:, i],
                action_head.mean_head.weight.data[:, i],
            )
        for i in expected_kept:
            assert torch.equal(
                expert.hidden[0].weight.data[i, :],
                action_head.trunk[0].weight.data[i, :],
            ), f"kept neuron {i} must equal ActionHead bit-exactly"
            assert torch.equal(
                expert.hidden[0].bias.data[i],
                action_head.trunk[0].bias.data[i],
            )
            assert torch.equal(
                expert.output_proj.weight.data[:, i],
                action_head.mean_head.weight.data[:, i],
            )

    rows_e0 = experts[0].hidden[0].weight.data[expected_dropped, :].clone()
    rows_e1 = experts[1].hidden[0].weight.data[expected_dropped, :].clone()
    assert not torch.equal(rows_e0, rows_e1), (
        "reinitialized neurons must differ across experts (per-expert draws)"
    )


def test_partial_reinit_is_seed_deterministic() -> None:
    _, action_head, _, _, _ = _make_components()

    def make_pair() -> nn.ModuleList:
        return nn.ModuleList(
            [ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7) for _ in range(2)]
        )

    experts_a = make_pair()
    experts_b = make_pair()
    experts_c = make_pair()

    torch.manual_seed(0)
    action_head.trunk[0].weight.data.normal_()
    action_head.trunk[0].bias.data.normal_()
    action_head.mean_head.weight.data.normal_()
    action_head.mean_head.bias.data.normal_()

    partial_reinit_experts_from_action_head(experts_a, action_head, drop_rate=0.5, seed=123)
    partial_reinit_experts_from_action_head(experts_b, action_head, drop_rate=0.5, seed=123)
    partial_reinit_experts_from_action_head(experts_c, action_head, drop_rate=0.5, seed=999)

    for idx in range(2):
        assert torch.equal(
            experts_a[idx].hidden[0].weight.data,
            experts_b[idx].hidden[0].weight.data,
        ), "same seed must yield bit-identical experts"
        assert not torch.equal(
            experts_a[idx].hidden[0].weight.data,
            experts_c[idx].hidden[0].weight.data,
        ), "different seeds must differ"


def test_partial_reinit_drop_rate_bounds() -> None:
    _, action_head, _, _, _ = _make_components()
    experts = nn.ModuleList(
        [ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7) for _ in range(2)]
    )
    torch.manual_seed(0)
    action_head.trunk[0].weight.data.normal_()
    action_head.trunk[0].bias.data.normal_()
    action_head.mean_head.weight.data.normal_()

    partial_reinit_experts_from_action_head(experts, action_head, drop_rate=0.0, seed=42)
    for expert in experts:
        assert torch.equal(expert.hidden[0].weight.data, action_head.trunk[0].weight.data)
        assert torch.equal(expert.output_proj.weight.data, action_head.mean_head.weight.data)

    partial_reinit_experts_from_action_head(experts, action_head, drop_rate=1.0, seed=42)
    for expert in experts:
        assert not torch.equal(expert.hidden[0].weight.data, action_head.trunk[0].weight.data)
        assert not torch.equal(expert.output_proj.weight.data, action_head.mean_head.weight.data)

    with pytest.raises(ValueError, match="drop_rate"):
        partial_reinit_experts_from_action_head(experts, action_head, drop_rate=-0.1, seed=42)
    with pytest.raises(ValueError, match="drop_rate"):
        partial_reinit_experts_from_action_head(experts, action_head, drop_rate=1.5, seed=42)


def test_one_warm_isolates_warm_expert() -> None:
    _, action_head, _, _, _ = _make_components()
    experts = nn.ModuleList(
        [ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7) for _ in range(3)]
    )
    torch.manual_seed(0)
    action_head.trunk[0].weight.data.normal_()
    action_head.trunk[0].bias.data.normal_()
    action_head.mean_head.weight.data.normal_()
    action_head.mean_head.bias.data.normal_()

    one_warm_experts_from_action_head(
        experts, action_head, jitter_std=0.0, warm_idx=1
    )

    assert torch.equal(experts[1].hidden[0].weight.data, action_head.trunk[0].weight.data)
    assert torch.equal(experts[1].hidden[0].bias.data, action_head.trunk[0].bias.data)
    assert torch.equal(experts[1].output_proj.weight.data, action_head.mean_head.weight.data)
    assert torch.equal(experts[1].output_proj.bias.data, action_head.mean_head.bias.data)

    assert not torch.equal(experts[0].hidden[0].weight.data, action_head.trunk[0].weight.data)
    assert not torch.equal(experts[2].hidden[0].weight.data, action_head.trunk[0].weight.data)
    assert not torch.equal(experts[0].hidden[0].weight.data, experts[2].hidden[0].weight.data)


def test_one_warm_rejects_out_of_bounds_idx() -> None:
    _, action_head, _, _, _ = _make_components()
    experts = nn.ModuleList(
        [ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7) for _ in range(3)]
    )
    with pytest.raises(ValueError, match="warm_idx"):
        one_warm_experts_from_action_head(experts, action_head, warm_idx=3)
    with pytest.raises(ValueError, match="warm_idx"):
        one_warm_experts_from_action_head(experts, action_head, warm_idx=-1)
    # Empty expert list is also invalid (silent no-op would hide misconfig).
    with pytest.raises(ValueError, match="warm_idx"):
        one_warm_experts_from_action_head(
            nn.ModuleList([]), action_head, warm_idx=0
        )


def test_bootstrap_dispatches_partial_warm_and_one_warm() -> None:
    for cfg in (
        {"type": "partial_warm", "drop_rate": 0.5, "seed": 42},
        {"type": "one_warm", "warm_idx": 0, "jitter_std": 0.0},
        {"type": "warmstart", "jitter_std": 0.02},
        {"type": "random"},
    ):
        encoder, action_head, router, expert, phase_head = _make_components()
        model = PhaseBootstrappedMoE(
            encoder=encoder,
            action_head=action_head,
            phase_head=phase_head,
            router=router,
            expert=expert,
            expert_init=cfg,
        )
        model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")
        assert model.stage == 2
        # Audit metadata must be populated for every dispatched init type.
        info = getattr(model, "_expert_init_info", None)
        assert info is not None, f"_expert_init_info missing for cfg={cfg}"
        assert info["expert_init"]["type"] == cfg["type"]
        assert info["router"]["num_experts"] == 4
        assert info["router"]["top_k"] == 2
        assert info["router"]["init_type"] == "centroid"


def test_bootstrap_rejects_unknown_expert_init_type() -> None:
    encoder, action_head, router, expert, phase_head = _make_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
        expert_init={"type": "not_a_real_type"},
    )
    with pytest.raises(ValueError, match="Unknown expert_init type"):
        model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")
    assert model.stage == 1


def test_expert_init_info_is_none_before_bootstrap() -> None:
    """The audit-metadata attribute is declared in __init__ as None.

    Bootstrap populates it; until then consumers (the cli metadata writer,
    tests, analysis scripts) should see ``None``, not a missing attribute.
    """
    encoder, action_head, router, expert, phase_head = _make_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
        expert_init={"type": "warmstart"},
    )
    assert hasattr(model, "_expert_init_info")
    assert model._expert_init_info is None

    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")
    assert model._expert_init_info is not None


def test_bootstrap_metadata_uses_resolved_router_override() -> None:
    """The recorded router init must reflect the bootstrap override, not the
    constructor default — otherwise direct ``bootstrap_moe(router_init=...)``
    calls (outside the manifest path) silently disagree with what was actually
    used.
    """
    encoder, action_head, router, expert, phase_head = _make_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
        expert_init={"type": "warmstart"},
    )
    model.bootstrap_moe(
        dataloader=_make_dataloader(),
        device="cpu",
        router_init={"type": "random", "seed": 7},
    )
    assert model._expert_init_info["router"]["init_type"] == "random"
    assert model._expert_init_info["router"]["init_seed"] == 7


def test_verify_bank_against_baseline_passes_on_match() -> None:
    """The g8 bank check passes when reset_bank matches the g6 baseline."""
    import json
    import sys
    import tempfile
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from experiments import v2_gates

    with tempfile.TemporaryDirectory() as tmpdir:
        findings_dir = Path(tmpdir) / "_findings"
        findings_dir.mkdir()
        # Baseline: g6 phaseforge_e6 on bank aaa / bbb / ccc per seed.
        (findings_dir / "v2_gates_g6.json").write_text(
            json.dumps(
                {
                    "gate": "g6",
                    "results": {
                        "phaseforge_e6:42": {"reset_bank": "aaa"},
                        "phaseforge_e6:43": {"reset_bank": "bbb"},
                        "phaseforge_e6:44": {"reset_bank": "ccc"},
                    },
                }
            ),
            encoding="utf-8",
        )
        original = v2_gates.FINDINGS_DIR
        v2_gates.FINDINGS_DIR = findings_dir
        try:
            spec = {
                "methods": ["warmstart_r50"],
                "baseline_gate": "g6",
                "baseline_method": "phaseforge_e6",
            }
            results = {
                "warmstart_r50:42": {"reset_bank": "aaa"},
                "warmstart_r50:43": {"reset_bank": "bbb"},
                "warmstart_r50:44": {"reset_bank": "ccc"},
            }
            ok, msg = v2_gates._verify_bank_against_baseline(
                "g8", [42, 43, 44], results, spec
            )
            assert ok, f"expected match, got error: {msg}"
            assert msg == ""
        finally:
            v2_gates.FINDINGS_DIR = original


def test_verify_bank_against_baseline_fails_on_mismatch() -> None:
    """The g8 bank check fails when a cell's reset_bank differs from g6."""
    import json
    import sys
    import tempfile
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from experiments import v2_gates

    with tempfile.TemporaryDirectory() as tmpdir:
        findings_dir = Path(tmpdir) / "_findings"
        findings_dir.mkdir()
        (findings_dir / "v2_gates_g6.json").write_text(
            json.dumps(
                {
                    "gate": "g6",
                    "results": {
                        "phaseforge_e6:42": {"reset_bank": "aaa"},
                        "phaseforge_e6:43": {"reset_bank": "bbb"},
                        "phaseforge_e6:44": {"reset_bank": "ccc"},
                    },
                }
            ),
            encoding="utf-8",
        )
        original = v2_gates.FINDINGS_DIR
        v2_gates.FINDINGS_DIR = findings_dir
        try:
            spec = {
                "methods": ["warmstart_r50"],
                "baseline_gate": "g6",
                "baseline_method": "phaseforge_e6",
            }
            results = {
                # seed 43 mismatch: cell on bank xxx, baseline on bbb.
                "warmstart_r50:42": {"reset_bank": "aaa"},
                "warmstart_r50:43": {"reset_bank": "xxx"},
                "warmstart_r50:44": {"reset_bank": "ccc"},
            }
            ok, msg = v2_gates._verify_bank_against_baseline(
                "g8", [42, 43, 44], results, spec
            )
            assert not ok
            assert "bank mismatch" in msg
            assert "warmstart_r50:43" in msg
            assert "'xxx'" in msg
            assert "'bbb'" in msg
        finally:
            v2_gates.FINDINGS_DIR = original


def test_verify_bank_against_baseline_skips_when_baseline_missing() -> None:
    """No g6 findings → skip the check with a warning, do not fail."""
    import sys
    import tempfile
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from experiments import v2_gates

    with tempfile.TemporaryDirectory() as tmpdir:
        findings_dir = Path(tmpdir) / "_findings"
        findings_dir.mkdir()
        original = v2_gates.FINDINGS_DIR
        v2_gates.FINDINGS_DIR = findings_dir
        try:
            spec = {
                "methods": ["warmstart_r50"],
                "baseline_gate": "g6",
                "baseline_method": "phaseforge_e6",
            }
            results = {
                "warmstart_r50:42": {"reset_bank": "aaa"},
            }
            ok, msg = v2_gates._verify_bank_against_baseline(
                "g8", [42], results, spec
            )
            assert ok, "missing baseline should not fail"
            assert msg == ""
        finally:
            v2_gates.FINDINGS_DIR = original


def test_partial_reinit_drop_rate_one_is_deterministic() -> None:
    _, action_head, _, _, _ = _make_components()

    def make() -> nn.ModuleList:
        return nn.ModuleList(
            [ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7) for _ in range(2)]
        )

    torch.manual_seed(0)
    action_head.trunk[0].weight.data.normal_()
    action_head.trunk[0].bias.data.normal_()
    action_head.mean_head.weight.data.normal_()
    action_head.mean_head.bias.data.normal_()

    a = make()
    b = make()
    c = make()
    partial_reinit_experts_from_action_head(a, action_head, drop_rate=1.0, seed=7)
    partial_reinit_experts_from_action_head(b, action_head, drop_rate=1.0, seed=7)
    partial_reinit_experts_from_action_head(c, action_head, drop_rate=1.0, seed=8)

    for idx in range(2):
        assert torch.equal(
            a[idx].hidden[0].weight.data, b[idx].hidden[0].weight.data
        ), "drop_rate=1.0 with same seed must be bit-identical"
        assert torch.equal(
            a[idx].output_proj.weight.data, b[idx].output_proj.weight.data
        )
        assert not torch.equal(
            a[idx].hidden[0].weight.data, c[idx].hidden[0].weight.data
        ), "drop_rate=1.0 with different seed must differ"

    dropped = partial_reinit_experts_from_action_head(
        make(), action_head, drop_rate=1.0, seed=7
    )
    assert dropped == list(range(64)), (
        "drop_rate=1.0 must drop every neuron and return all indices"
    )


def test_partial_reinit_returned_indices_match_dropped_set() -> None:
    _, action_head, _, _, _ = _make_components()
    experts = nn.ModuleList(
        [ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7) for _ in range(2)]
    )
    torch.manual_seed(0)
    action_head.trunk[0].weight.data.normal_()
    action_head.mean_head.weight.data.normal_()

    dropped = partial_reinit_experts_from_action_head(
        experts, action_head, drop_rate=0.3, seed=2026
    )
    expected = sorted(
        torch.randperm(64, generator=torch.Generator().manual_seed(2026))[
            : int(round(0.3 * 64))
        ].tolist()
    )
    assert dropped == expected
    assert len(dropped) == 19
    assert len(set(dropped)) == 19


def test_bootstrap_persists_partial_warm_metadata() -> None:
    import hashlib

    encoder, action_head, router, expert, phase_head = _make_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
        expert_init={"type": "partial_warm", "drop_rate": 0.5, "seed": 42},
    )
    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")

    info = model._expert_init_info
    assert info["expert_init"]["type"] == "partial_warm"
    assert info["expert_init"]["drop_rate"] == 0.5
    assert info["expert_init"]["init_seed"] == 42
    assert info["expert_init"]["hidden_dim"] == 64
    assert info["expert_init"]["num_dropped_neurons"] == 32
    assert len(info["expert_init"]["dropped_neuron_indices"]) == 32
    # Verify hash consistency: sha256 of sorted indices as 8-byte little-endian.
    h = hashlib.sha256()
    for i in sorted(info["expert_init"]["dropped_neuron_indices"]):
        h.update(int(i).to_bytes(8, "little", signed=False))
    assert info["expert_init"]["dropped_indices_sha256"] == h.hexdigest()
    assert info["router"]["init_type"] == "centroid"
    assert info["router"]["num_experts"] == 4


def test_bootstrap_rotates_one_warm_by_training_seed() -> None:
    encoder, action_head, router, expert, phase_head = _make_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
        expert_init={
            "type": "one_warm",
            "warm_idx": 0,
            "jitter_std": 0.0,
            "rotate_warm_idx_by_seed": True,
        },
    )
    torch.manual_seed(0)
    action_head.trunk[0].weight.data.normal_()
    action_head.trunk[0].bias.data.normal_()
    action_head.mean_head.weight.data.normal_()
    action_head.mean_head.bias.data.normal_()

    model.bootstrap_moe(
        dataloader=_make_dataloader(),
        device="cpu",
        training_seed=2,
    )
    info = model._expert_init_info
    # 4 experts, requested warm_idx=0, training_seed=2 -> effective (0+2)%4 = 2.
    assert info["expert_init"]["warm_idx"] == 2
    assert info["expert_init"]["requested_warm_idx"] == 0
    assert info["expert_init"]["rotate_warm_idx_by_seed"] is True
    assert info["expert_init"]["training_seed"] == 2
    # Expert 2 should equal ActionHead bit-exactly (jitter_std=0.0).
    assert torch.equal(
        model.moe_layer.experts[2].hidden[0].weight.data,
        action_head.trunk[0].weight.data,
    )
    # Other experts are reset (different from ActionHead).
    assert not torch.equal(
        model.moe_layer.experts[0].hidden[0].weight.data,
        action_head.trunk[0].weight.data,
    )


def test_bootstrap_rejects_one_warm_seed_rotation_without_seed() -> None:
    encoder, action_head, router, expert, phase_head = _make_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
        expert_init={
            "type": "one_warm",
            "rotate_warm_idx_by_seed": True,
        },
    )
    with pytest.raises(ValueError, match="training_seed"):
        model.bootstrap_moe(
            dataloader=_make_dataloader(),
            device="cpu",
            training_seed=None,
        )
    assert model.stage == 1


def test_manifest_has_wave3_cells_with_correct_overrides() -> None:
    """The Wave-3 cells must override the 8-expert / soft_mapping defaults.

    Without these overrides the comparison against ``pf_centroid_random`` (6
    experts, centroid) would confound expert initialization with the V2-B
    config change. See the reviewer's P0 note.
    """
    import json
    from pathlib import Path

    manifest_path = Path(__file__).resolve().parents[2] / "experiments" / "lift_ablation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {m["name"]: m for m in manifest["methods"]}

    r50 = by_name["warmstart_r50"]
    assert r50["model"] == "phaseforge", "warmstart_r50 should build on phaseforge"
    r50_overrides = " ".join(r50["overrides"])
    assert "models.router.num_experts=6" in r50_overrides, (
        "warmstart_r50 must pin num_experts=6 to match pf_centroid_random"
    )
    assert "models.router_init.type=centroid" in r50_overrides, (
        "warmstart_r50 must pin router_init.type=centroid (NOT the "
        "soft_mapping default in phaseforge.yaml)"
    )
    assert "models.expert_init.type=partial_warm" in r50_overrides
    assert "+models.expert_init.drop_rate=0.5" in r50_overrides
    assert "+models.expert_init.seed=${project.seed}" in r50_overrides, (
        "warmstart_r50 should tie init seed to training seed for "
        "seed-robustness studies"
    )

    onewarm = by_name["pf_one_warm_plus_random"]
    onewarm_overrides = " ".join(onewarm["overrides"])
    assert "models.router.num_experts=6" in onewarm_overrides
    assert "models.router_init.type=centroid" in onewarm_overrides
    assert "models.expert_init.type=one_warm" in onewarm_overrides
    assert "+models.expert_init.rotate_warm_idx_by_seed=true" in onewarm_overrides, (
        "pf_one_warm_plus_random must rotate warm_idx by training seed "
        "to avoid the phase-0 confound the reviewer flagged"
    )


def test_canonical_phaseforge_config_resolves_to_r50_contract() -> None:
    """The canonical ``phaseforge`` config must resolve to the R50 contract.

    Permanent guard for the final migration: the canonical method identity is
    the promoted R50 implementation (six experts, top-2 routing, centroid
    router init, seed-dependent 50% partial warm-start, soft mapping off,
    frozen Stage 2 encoder), and the separate ``phaseforge_r50`` config no
    longer exists. The pre-final 8-expert / soft-mapping / warmstart-with-
    jitter defaults must never silently return.
    """
    from hydra import compose, initialize

    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        cfg42 = compose(
            config_name="main",
            overrides=[
                "models=phaseforge",
                "train=stage2",
                "data=common",
                "project.seed=42",
            ],
        )
        cfg43 = compose(
            config_name="main",
            overrides=[
                "models=phaseforge",
                "train=stage2",
                "data=common",
                "project.seed=43",
            ],
        )

    for cfg, seed in ((cfg42, 42), (cfg43, 43)):
        m = cfg.models
        assert m.name == "phaseforge"
        assert m._target_ == "phaseforge.models.phase_moe.PhaseBootstrappedMoE"
        assert m.router.num_experts == 6, "must not reintroduce num_experts=8"
        assert m.router.top_k == 2
        assert m.router.normalize_input is True
        assert m.router_init.type == "centroid", "must not reintroduce soft_mapping"
        assert m.expert_init.type == "partial_warm", "must not be plain warmstart"
        assert m.expert_init.drop_rate == 0.5
        assert m.expert_init.jitter_std == 0.0
        assert m.expert_init.seed == seed, (
            "partial-init seed must follow the training seed, not stay constant"
        )
        assert m.soft_mapping.enabled is False, "soft-mapping machinery must be off"
        assert cfg.train.freeze_encoder is True, "Stage 2 must freeze the encoder"

    # The absorbed R50 config must not survive as a separate active config:
    # resolving it must fail loudly rather than fall back to anything.
    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        try:
            compose(
                config_name="main",
                overrides=["models=phaseforge_r50"],
            )
        except Exception as exc:  # hydra.errors.MissingConfigException et al.
            assert "phaseforge_r50" in str(exc)
        else:
            raise AssertionError(
                "models=phaseforge_r50 must no longer resolve after the "
                "canonical migration; the config file was deleted."
            )



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


# ---------------------------------------------------------------------------
# V2-B: soft phase->expert mapping bootstrap
# ---------------------------------------------------------------------------


def _make_soft_mapping_components(num_experts: int = 8):
    encoder = StateEncoder(input_dim=151, hidden_dims=[64], latent_dim=32)
    action_head = ActionHead(input_dim=32, output_dim=7, hidden_dim=64)
    router = TopKRouter(latent_dim=32, num_experts=num_experts, top_k=2)
    expert = ExpertMLP(input_dim=32, hidden_dims=[64], output_dim=7)
    phase_head = PhaseClassificationHead(latent_dim=32, num_phases=3)
    return encoder, action_head, router, expert, phase_head


def test_soft_mapping_buffer_is_zero_init_and_persistent() -> None:
    encoder, action_head, router, expert, phase_head = _make_soft_mapping_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
    )
    assert model.soft_mapping.shape == (3, 8)
    assert not model.soft_mapping.any()
    assert "soft_mapping" in model.state_dict()


def test_require_soft_mapping_fails_closed_when_zero() -> None:
    encoder, action_head, router, expert, phase_head = _make_soft_mapping_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
    )
    with pytest.raises(RuntimeError, match="all-zero"):
        model.require_soft_mapping()


def test_bootstrap_hierarchical_uniform_fills_buffer_and_keeps_random_gate() -> None:
    encoder, action_head, router, expert, phase_head = _make_soft_mapping_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
        router_init={
            "type": "soft_mapping",
            "mapping_mode": "hierarchical_uniform",
            "seed": 42,
        },
    )
    gate_before = model.moe_layer.router.gate_linear.weight.data.clone()

    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")

    assert model.stage == 2
    expected = build_hierarchical_uniform_mapping(3, 8)
    assert torch.allclose(model.soft_mapping, expected)
    # Data-free mode must not touch the (random) gate weights.
    assert torch.allclose(model.moe_layer.router.gate_linear.weight.data, gate_before)
    # The teacher/oracle paths can now consume the mapping.
    assert model.require_soft_mapping() is model.soft_mapping


def test_bootstrap_prototype_softmax_sets_gate_and_mapping() -> None:
    encoder, action_head, router, expert, phase_head = _make_soft_mapping_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
        router_init={
            "type": "soft_mapping",
            "mapping_mode": "prototype_softmax",
            "temperature": 1.0,
            "seed": 42,
        },
    )
    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")

    assert model.stage == 2
    mapping = model.soft_mapping
    assert mapping.shape == (3, 8)
    assert torch.allclose(mapping.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert mapping.any()
    # Gate rows are the unit-norm hierarchical prototypes (cosine geometry).
    rows = model.moe_layer.router.gate_linear.weight.data
    assert torch.allclose(rows.norm(dim=-1), torch.ones(8), atol=1e-5)


def test_bootstrap_unknown_soft_mapping_mode_raises() -> None:
    encoder, action_head, router, expert, phase_head = _make_soft_mapping_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
        router_init={"type": "soft_mapping", "mapping_mode": "mystery"},
    )
    with pytest.raises(ValueError, match="mapping_mode"):
        model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")
    assert model.stage == 1


# ---------------------------------------------------------------------------
# V2-D: teacher-routed KL path
# ---------------------------------------------------------------------------


def test_teacher_routing_off_by_default_no_phase_logits() -> None:
    encoder, action_head, router, expert, phase_head = _make_soft_mapping_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
    )
    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")
    assert not model.teacher_routing_enabled
    out = model({"state": torch.randn(2, 151)})
    assert out.phase_logits is None


def test_teacher_routing_on_emits_phase_logits_in_stage2() -> None:
    encoder, action_head, router, expert, phase_head = _make_soft_mapping_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
        teacher_routing={"enabled": True},
        router_init={
            "type": "soft_mapping",
            "mapping_mode": "hierarchical_uniform",
            "seed": 42,
        },
    )
    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")
    assert model.teacher_routing_enabled
    out = model({"state": torch.randn(2, 151)})
    assert out.phase_logits is not None
    assert out.phase_logits.shape == (2, 3)


def test_teacher_routing_on_fails_closed_on_zero_mapping() -> None:
    # Stage 2 without bootstrap (or a pre-V2-B checkpoint): M is all zero and
    # the teacher path must refuse to run rather than distill through zeros.
    encoder, action_head, router, expert, phase_head = _make_soft_mapping_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
        teacher_routing={"enabled": True},
    )
    model.stage = 2
    with pytest.raises(RuntimeError, match="all-zero"):
        model({"state": torch.randn(2, 151)})


# ---------------------------------------------------------------------------
# V2-E: evaluation-time routing interventions
# ---------------------------------------------------------------------------


def _bootstrapped_model(num_experts: int = 8) -> PhaseBootstrappedMoE:
    encoder, action_head, router, expert, phase_head = _make_soft_mapping_components(
        num_experts
    )
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
        router_init={
            "type": "soft_mapping",
            "mapping_mode": "hierarchical_uniform",
            "seed": 42,
        },
    )
    model.bootstrap_moe(dataloader=_make_dataloader(), device="cpu")
    return model


def test_eval_mode_default_and_validation() -> None:
    model = _bootstrapped_model()
    assert model.eval_mode == "learned"
    with pytest.raises(ValueError, match="Unknown eval_mode"):
        model.eval_mode = "mystery"
    for mode in ("learned", "sticky", "uniform", "oracle"):
        model.eval_mode = mode
        assert model.eval_mode == mode
    model.eval_mode = "LEARNED"
    assert model.eval_mode == "learned"


def test_eval_modes_override_dispatch_but_keep_gate_logits() -> None:
    model = _bootstrapped_model(num_experts=4)
    state = torch.randn(2, 151)
    learned = model({"state": state})
    assert learned.gate_logits is not None

    # Uniform mode: every expert is selected with equal weight.
    model.eval_mode = "uniform"
    uniform = model({"state": state})
    assert uniform.expert_indices.shape == (2, 4)
    assert torch.allclose(
        uniform.routing_weights, torch.full((2, 4), 0.25)
    )
    # Gate logits stay the learned ones (reported, not used).
    assert torch.equal(uniform.gate_logits, learned.gate_logits)

    # Oracle mode: M^T softmax(phase_head(z)) selects the phase's experts.
    model.eval_mode = "oracle"
    oracle = model({"state": state})
    assert oracle.expert_indices.shape == (2, 2)
    mapping = model.require_soft_mapping()
    phase_probs = torch.softmax(model.phase_head(model.encoder(state)), dim=-1)
    expected = torch.topk(
        torch.einsum("pe,bp->be", mapping, phase_probs), 2, dim=-1
    ).indices
    assert torch.equal(oracle.expert_indices, expected)


def test_sticky_eval_mode_resets_per_episode() -> None:
    model = _bootstrapped_model(num_experts=4)
    model.eval_mode = "sticky"
    router = model.moe_layer.router
    assert router._sticky_ema is None

    # First step initializes the EMA; later steps keep updating it.
    model.get_action(torch.randn(1, 151))
    assert router._sticky_ema is not None
    ema_after_first = router._sticky_ema.clone()
    model.get_action(torch.randn(1, 151))
    assert not torch.equal(router._sticky_ema, ema_after_first)

    # reset() clears the EMA for the next episode (the runner calls it).
    model.reset()
    assert router._sticky_ema is None


def test_oracle_eval_mode_fails_closed_on_zero_mapping() -> None:
    encoder, action_head, router, expert, phase_head = _make_soft_mapping_components()
    model = PhaseBootstrappedMoE(
        encoder=encoder,
        action_head=action_head,
        phase_head=phase_head,
        router=router,
        expert=expert,
    )
    model.stage = 2
    model.eval_mode = "oracle"
    with pytest.raises(RuntimeError, match="all-zero"):
        model.get_action(torch.randn(1, 151))


def test_unknown_router_override_rejected_in_moe_layer() -> None:
    model = _bootstrapped_model()
    with pytest.raises(ValueError, match="Unknown router_override"):
        model.moe_layer(
            torch.randn(2, 32),
            router_override="mystery",
        )
