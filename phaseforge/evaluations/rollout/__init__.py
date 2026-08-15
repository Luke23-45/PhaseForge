"""Rollout protocol: frozen reset bank, simulator adapter, runners, gates."""

from phaseforge.evaluations.rollout.report import build_rollout_report
from phaseforge.evaluations.rollout.reset_bank import (
    ResetBank,
    ResetCase,
    bank_dir,
    compute_bank_id,
    generate_reset_bank,
)
from phaseforge.evaluations.rollout.runner import (
    RolloutEvaluator,
    RolloutOutcome,
    RolloutRunInvalid,
    resolve_pinned_metadata,
    run_rollout_evaluation,
)
from phaseforge.evaluations.rollout.scripted_controller import (
    ScriptedCanController,
    ScriptedController,
    ScriptedControllerConfig,
    ScriptedLiftConfig,
    ScriptedLiftController,
    ScriptedSquareController,
    ScriptedToolHangController,
    ScriptedTransportController,
)

__all__ = [
    "ResetBank",
    "ResetCase",
    "ScriptedController",
    "ScriptedLiftController",
    "ScriptedCanController",
    "ScriptedSquareController",
    "ScriptedToolHangController",
    "ScriptedTransportController",
    "ScriptedControllerConfig",
    "ScriptedLiftConfig",
    "RolloutEvaluator",
    "RolloutOutcome",
    "RolloutRunInvalid",
    "bank_dir",
    "build_rollout_report",
    "compute_bank_id",
    "generate_reset_bank",
    "resolve_pinned_metadata",
    "run_rollout_evaluation",
]
