"""CPU-only tests for the CLI module's strict checkpoint-loading diagnostics.

``phaseforge.cli`` must stay importable without ``wandb`` installed (the
import is lazy), and checkpoint loads must hard-fail on missing/unexpected
weights instead of silently continuing on random weights — a mismatched
eval model config otherwise runs on random weights and produces meaningless
rollout results with no error. The only allowed skip is the Stage 1 -> Stage 2
bootstrap (the ``moe_layer`` prefix), which is logged at INFO.
"""

from __future__ import annotations

import logging

import pytest
import torch

import phaseforge.cli as cli


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 2)


def test_cli_importable_without_wandb() -> None:
    """The module imports even though wandb is not installed (lazy import)."""
    # This test env intentionally lacks wandb; the import above already proves
    # the module-level import works. Guard against a future regression by
    # asserting the module is not None and wandb is not imported at top level.
    assert cli is not None
    assert "wandb" not in getattr(cli, "__dict__", {})


def test_load_state_dict_checked_passes_on_perfect_match() -> None:
    model = _DummyModel()
    cli._load_state_dict_checked(
        model, model.state_dict(), "Evaluation checkpoint load"
    )


def test_load_state_dict_checked_raises_on_missing_keys() -> None:
    model = _DummyModel()
    state_dict = model.state_dict()
    del state_dict["head.weight"]
    with pytest.raises(RuntimeError, match="1 missing"):
        cli._load_state_dict_checked(
            model, state_dict, "Evaluation checkpoint load"
        )


def test_load_state_dict_checked_raises_on_unexpected_keys() -> None:
    model = _DummyModel()
    state_dict = dict(model.state_dict())
    state_dict["moe_layer.router.gate_linear.weight"] = torch.zeros(4, 4)
    with pytest.raises(RuntimeError, match="unexpected"):
        cli._load_state_dict_checked(
            model, state_dict, "Evaluation checkpoint load"
        )


def test_load_state_dict_checked_allows_expected_prefix_at_info(caplog) -> None:
    """The Stage 1 -> Stage 2 bootstrap legitimately skips the MoE block."""
    model = _DummyModel()
    state_dict = dict(model.state_dict())
    state_dict["moe_layer.router.gate_linear.weight"] = torch.zeros(4, 4)
    state_dict["moe_layer.experts.0.hidden.0.weight"] = torch.zeros(4, 4)

    with caplog.at_level(logging.INFO, logger="phaseforge.cli"):
        cli._load_state_dict_checked(
            model, state_dict, "Stage 1 -> Stage 2 bootstrap load",
            expected_unexpected_prefixes=("moe_layer",),
        )

    assert "expected, skipped" in caplog.text
    assert "2 key(s) differ" in caplog.text


def test_load_state_dict_checked_allows_expected_missing_prefix_at_info(caplog) -> None:
    """A BC checkpoint legitimately lacks the target MoE block."""
    model = _DummyModel()
    model.moe_layer = torch.nn.Linear(4, 4)
    state_dict = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("moe_layer")
    }

    with caplog.at_level(logging.INFO, logger="phaseforge.cli"):
        cli._load_state_dict_checked(
            model, state_dict, "Stage 1 -> Stage 2 bootstrap load",
            expected_unexpected_prefixes=("moe_layer",),
        )

    assert "2 key(s) differ" in caplog.text


def test_load_state_dict_checked_raises_when_unknown_keys_remain() -> None:
    """Unexpected keys OUTSIDE the expected prefixes must still fail."""
    model = _DummyModel()
    state_dict = dict(model.state_dict())
    state_dict["moe_layer.router.gate_linear.weight"] = torch.zeros(4, 4)
    state_dict["encoder.extra_bias"] = torch.zeros(4)

    with pytest.raises(RuntimeError, match="encoder.extra_bias"):
        cli._load_state_dict_checked(
            model, state_dict, "Stage 1 -> Stage 2 bootstrap load",
            expected_unexpected_prefixes=("moe_layer",),
        )


def _build_model_for_test(name: str) -> torch.nn.Module:
    from omegaconf import DictConfig, OmegaConf

    from phaseforge.utils.registry import build_model

    path = (
        "phaseforge/config/models/phaseforge.yaml"
        if name == "phaseforge"
        else f"phaseforge/config/models/baselines/{name}.yaml"
    )
    # Model configs interpolate ${data.state_dim}/${data.action_dim}, so the
    # data block must be present for the config to resolve — same as the CLI.
    data_cfg = OmegaConf.load("phaseforge/config/data/common.yaml")
    return build_model(
        DictConfig({"models": OmegaConf.load(path), "data": data_cfg})
    )


def test_stage1_bootstrap_load_matrix_all_cells() -> None:
    """Every real Stage 1 -> Stage 2 source/target pair must load cleanly.

    Cells whose Stage 1 comes from ``bc`` (no phase head) or from the
    phaseforge cell must be loadable through the cli's checked bootstrap
    load, and the load must still reject a missing head the target NEEDS.
    """
    # A BC checkpoint (encoder + action head only) never has phase_head.
    bc = _build_model_for_test("bc")
    bc_sd = bc.state_dict()

    # The phaseforge cell's Stage 1 checkpoint includes the phase head.
    pf = _build_model_for_test("phaseforge")
    pf_sd = pf.state_dict()
    assert any(k.startswith("phase_head") for k in pf_sd)

    # warmstart_moe / plain_encoder <- bc: no phase head in either side.
    for name in ("warmstart_moe", "plain_encoder_phase_bootstrap"):
        model = _build_model_for_test(name)
        assert not any(k.startswith("phase_head") for k in model.state_dict())
        cli._load_state_dict_checked(
            model, bc_sd, f"{name} <- bc bootstrap load",
            expected_unexpected_prefixes=("moe_layer",),
        )

    # phaseforge / teacher_forced <- phaseforge: identical head structure.
    for name in ("phaseforge", "teacher_forced"):
        model = _build_model_for_test(name)
        assert any(k.startswith("phase_head") for k in model.state_dict())
        cli._load_state_dict_checked(
            model, pf_sd, f"{name} <- phaseforge bootstrap load",
            expected_unexpected_prefixes=("moe_layer",),
        )

    # phase_pretrain_random_router <- phaseforge: no phase_head module in
    # the target, so the checkpoint's phase_head keys are unused heads and
    # may be dropped; anything else must still fail.
    model = _build_model_for_test("phase_pretrain_random_router")
    assert not any(k.startswith("phase_head") for k in model.state_dict())
    cli._load_state_dict_checked(
        model, pf_sd, "phase_pretrain_random_router <- phaseforge bootstrap load",
        expected_unexpected_prefixes=("moe_layer", "phase_head"),
    )


def test_unused_stage1_head_prefixes_are_target_specific() -> None:
    """The cli derives droppable Stage 1 heads from the TARGET architecture.

    A cell without a phase head may drop the checkpoint's phase_head keys;
    a cell that routes by the phase head may not.
    """
    assert cli._unused_stage1_head_prefixes(
        _build_model_for_test("phase_pretrain_random_router")
    ) == ("phase_head",)
    assert cli._unused_stage1_head_prefixes(
        _build_model_for_test("warmstart_moe")
    ) == ("phase_head",)
    assert cli._unused_stage1_head_prefixes(
        _build_model_for_test("plain_encoder_phase_bootstrap")
    ) == ("phase_head",)
    assert cli._unused_stage1_head_prefixes(
        _build_model_for_test("teacher_forced")
    ) == ()
    assert cli._unused_stage1_head_prefixes(
        _build_model_for_test("phaseforge")
    ) == ()


def test_phase_head_required_when_target_uses_it() -> None:
    """A checkpoint missing the phase head must still fail for cells that
    route by it (the prefix is only allowed for heads the target lacks)."""
    teacher_forced = _build_model_for_test("teacher_forced")
    bc_sd = dict(_build_model_for_test("bc").state_dict())
    assert not any(k.startswith("phase_head") for k in bc_sd)

    with pytest.raises(RuntimeError, match="phase_head"):
        cli._load_state_dict_checked(
            teacher_forced, bc_sd, "teacher_forced <- bc bootstrap load",
            expected_unexpected_prefixes=("moe_layer",),
        )
