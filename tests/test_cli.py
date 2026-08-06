"""CPU-only tests for the CLI module's checkpoint-load diagnostics.

``phaseforge.cli`` must stay importable without ``wandb`` installed (the
import is lazy), and the ``strict=False`` checkpoint loads must surface
silently-skipped weights instead of hiding them — a mismatched eval model
config otherwise runs on random weights and scores 0% in LIBERO rollouts
with no error.
"""

from __future__ import annotations

import logging

from torch.nn.modules.module import _IncompatibleKeys

import phaseforge.cli as cli


def test_cli_importable_without_wandb() -> None:
    """The module imports even though wandb is not installed (lazy import)."""
    # This test env intentionally lacks wandb; the import above already proves
    # the module-level import works. Guard against a future regression by
    # asserting the module is not None and wandb is not imported at top level.
    assert cli is not None
    assert "wandb" not in getattr(cli, "__dict__", {})


def test_log_state_dict_mismatch_warns_on_skipped_keys(caplog) -> None:
    result = _IncompatibleKeys(missing_keys=["encoder.w1"], unexpected_keys=["moe_layer.router"])
    with caplog.at_level(logging.WARNING, logger="phaseforge.cli"):
        cli._log_state_dict_mismatch(result, "Evaluation checkpoint load")

    assert "checkpoint/model mismatch" in caplog.text
    assert "1 missing" in caplog.text
    assert "encoder.w1" in caplog.text
    assert "moe_layer.router" in caplog.text
    assert "0% success in LIBERO rollouts" in caplog.text


def test_log_state_dict_mismatch_silent_on_perfect_match(caplog) -> None:
    result = _IncompatibleKeys(missing_keys=[], unexpected_keys=[])
    with caplog.at_level(logging.INFO, logger="phaseforge.cli"):
        cli._log_state_dict_mismatch(result, "Evaluation checkpoint load")
    assert caplog.text == ""


def test_log_state_dict_mismatch_expected_prefixes_at_info(caplog) -> None:
    """The Stage 1 -> Stage 2 bootstrap legitimately skips the MoE block."""
    result = _IncompatibleKeys(
        missing_keys=[], unexpected_keys=["moe_layer.router.w", "moe_layer.experts.0.net.0"]
    )
    with caplog.at_level(logging.INFO, logger="phaseforge.cli"):
        cli._log_state_dict_mismatch(
            result, "Stage 1 -> Stage 2 bootstrap load",
            expected_unexpected_prefixes=("moe_layer",),
        )

    assert "expected, skipped" in caplog.text
    assert "checkpoint/model mismatch" not in caplog.text


def test_log_state_dict_mismatch_warns_when_unknown_keys_remain(caplog) -> None:
    """Unexpected keys OUTSIDE the expected prefixes must still warn."""
    result = _IncompatibleKeys(
        missing_keys=[], unexpected_keys=["moe_layer.router.w", "encoder.w2"]
    )
    with caplog.at_level(logging.WARNING, logger="phaseforge.cli"):
        cli._log_state_dict_mismatch(
            result, "Stage 1 -> Stage 2 bootstrap load",
            expected_unexpected_prefixes=("moe_layer",),
        )

    assert "checkpoint/model mismatch" in caplog.text
    assert "encoder.w2" in caplog.text
