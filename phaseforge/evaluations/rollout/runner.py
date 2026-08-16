"""Rollout evaluation runner (implementation plan §4.4).

Drives one evaluation run over the frozen reset bank with the strict
metric: infrastructure failures invalidate the run; policy failures
(NaN/invalid actions, policy exceptions) are valid failed episodes with a
``failure_category``. Every episode is appended to ``episodes.jsonl`` via
the schema validator, and per-run summary statistics land in
``rollout_summary.json`` plus the results dict the CLI writes to
``eval_results.json``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from phaseforge.data.common.normalizer import FrozenNormalizer
from phaseforge.evaluations.envs.env_metadata import (
    PinnedEnvMetadata,
    env_metadata_from_cache,
    verify_environment_parity,
)
from phaseforge.evaluations.envs.errors import (
    EnvParityError,
    InfrastructureError,
    PolicyInvalidActionError,
)
from phaseforge.evaluations.envs.robosuite_adapter import (
    RobosuiteStateAdapter,
    StateSpec,
)
from phaseforge.evaluations.rollout.reset_bank import (
    ResetBank,
    bank_dir,
    compute_bank_id,
    generate_reset_bank,
)
from phaseforge.outputs_writer.episodes import (
    append_episode_record,
    summarize_episodes,
    wilson_interval,
)

logger = logging.getLogger(__name__)

#: Failure categories emitted by the runner (frozen taxonomy, plan §4.4).
FAILURE_TIMEOUT = "task_timeout"
FAILURE_POLICY_INVALID_ACTION = "policy_invalid_action"
FAILURE_POLICY_EXCEPTION = "policy_exception"


class RolloutRunInvalid(RuntimeError):
    """Raised when the run contains infrastructure-failure episodes.

    The run is marked failed in the ledger; the episode rows remain on
    disk (append-only, audit-ready) and the run must be rerun.
    """


@dataclass
class RolloutOutcome:
    """Outcome of one episode, before it is turned into a record row."""

    valid: bool
    success: bool = False
    timed_out: bool = False
    steps: int = 0
    termination_reason: str | None = None
    failure_category: str | None = None
    exception: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class RolloutEvaluator:
    """Runs one policy (or the scripted controller) over the reset bank."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        policy: Callable[[np.ndarray], np.ndarray] | None,
        adapter: RobosuiteStateAdapter,
        bank: ResetBank,
        normalizer: FrozenNormalizer | None,
        model: torch.nn.Module | None,
        output_dir: str | Path,
        run_id: str,
        model_name: str,
        training_seed: int,
        task: str,
        checkpoint_sha256: str,
        tag: str | None = None,
        horizon: int | None = None,
        action_tolerance: float = 1e-4,
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self.adapter = adapter
        self.bank = bank
        self.normalizer = normalizer
        self.model = model
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.model_name = model_name
        self.training_seed = int(training_seed)
        self.task = task
        self.checkpoint_sha256 = checkpoint_sha256
        self.tag = tag
        self.horizon = int(horizon or adapter.horizon)
        self.action_tolerance = float(action_tolerance)

        # These values are invariant for the whole rollout. Resolving them
        # once avoids a parameter walk, an eval-mode traversal, and repeated
        # normalizer device transfers at every simulator step.
        self._model_device: torch.device | None = None
        self._normalizer_mean: torch.Tensor | None = None
        self._normalizer_std: torch.Tensor | None = None
        if self.model is not None:
            self.model.eval()
            try:
                self._model_device = next(self.model.parameters()).device
            except (StopIteration, AttributeError):
                self._model_device = torch.device("cpu")
            if self.normalizer is not None:
                self._normalizer_mean = self.normalizer.mean.to(self._model_device)
                self._normalizer_std = self.normalizer.std.to(self._model_device)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Evaluate every bank case; returns the results dict for eval_results.json.

        Raises:
            RolloutRunInvalid: any episode was an infrastructure failure
                (invalid episode rows stay on disk for audit).
        """
        rows: list[dict[str, Any]] = []
        had_infrastructure_failure = False

        for case in self.bank.cases:
            outcome, row = self._run_episode(case)
            append_episode_record(self.output_dir, row)
            rows.append(row)
            if not outcome.valid:
                had_infrastructure_failure = True
            logger.info(
                "case %02d: valid=%s success=%s steps=%d reason=%s",
                case.index,
                outcome.valid,
                outcome.success,
                outcome.steps,
                outcome.termination_reason,
            )

        results = self._summarize(rows)
        self._write_run_summary(results, rows)

        if had_infrastructure_failure:
            raise RolloutRunInvalid(
                f"Rollout run {self.run_id} contains infrastructure-failure "
                "episodes (see episodes.jsonl). The run is invalid and must "
                "be rerun; no success rate is reported from it."
            )
        return results

    # ------------------------------------------------------------------
    # Per-episode execution
    # ------------------------------------------------------------------

    def _run_episode(self, case) -> tuple[RolloutOutcome, dict[str, Any]]:
        """Run one bank case; returns ``(outcome, schema-valid record row)``."""
        try:
            state = self.adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
        except InfrastructureError as exc:
            return self._episode(case, RolloutOutcome(valid=False, exception=str(exc)))
        except Exception as exc:  # noqa: BLE001 — simulator misuse is infra
            return self._episode(case, RolloutOutcome(valid=False, exception=str(exc)))

        # Both externally supplied policies and stateful model objects may
        # carry episode history. Reset each independently; swallowing a
        # broken reset would leak one episode into the next and invalidate a
        # recurrent baseline, so surface it as a policy failure.
        reset_targets = []
        if self.policy is not None:
            reset_targets.append(("policy", self.policy))
        if self.model is not None and self.model is not self.policy:
            reset_targets.append(("model", self.model))
        for label, target in reset_targets:
            reset = getattr(target, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception as exc:  # noqa: BLE001 — policy state bug
                    return self._episode(
                        case,
                        RolloutOutcome(
                            valid=True,
                            steps=0,
                            termination_reason=FAILURE_POLICY_EXCEPTION,
                            failure_category=FAILURE_POLICY_EXCEPTION,
                            exception=f"{label}.reset failed: {type(exc).__name__}: {exc}",
                        ),
                    )

        steps = 0
        while steps < self.horizon:
            try:
                if self.policy is not None:
                    action = self.policy(state)
                else:
                    action = self._policy_action(state)
            except PolicyInvalidActionError as exc:
                outcome = RolloutOutcome(
                    valid=True,
                    steps=steps,
                    termination_reason=FAILURE_POLICY_INVALID_ACTION,
                    failure_category=FAILURE_POLICY_INVALID_ACTION,
                    exception=str(exc),
                )
                return self._episode(case, outcome)
            except Exception as exc:  # noqa: BLE001 — model raised
                outcome = RolloutOutcome(
                    valid=True,
                    steps=steps,
                    termination_reason=FAILURE_POLICY_EXCEPTION,
                    failure_category=FAILURE_POLICY_EXCEPTION,
                    exception=f"{type(exc).__name__}: {exc}",
                )
                return self._episode(case, outcome)

            try:
                state, _done, success, _info = self.adapter.step(action)
            except PolicyInvalidActionError as exc:
                return self._episode(
                    case,
                    RolloutOutcome(
                        valid=True,
                        steps=steps,
                        termination_reason=FAILURE_POLICY_INVALID_ACTION,
                        failure_category=FAILURE_POLICY_INVALID_ACTION,
                        exception=str(exc),
                    ),
                )
            except InfrastructureError as exc:
                return self._episode(case, RolloutOutcome(valid=False, exception=str(exc)))
            except Exception as exc:  # noqa: BLE001 — simulator misuse
                return self._episode(case, RolloutOutcome(valid=False, exception=str(exc)))
            steps += 1
            if success:
                return self._episode(
                    case,
                    RolloutOutcome(
                        valid=True,
                        success=True,
                        steps=steps,
                        termination_reason="success",
                    ),
                )

        return self._episode(
            case,
            RolloutOutcome(
                valid=True,
                timed_out=True,
                steps=steps,
                termination_reason=FAILURE_TIMEOUT,
                failure_category=FAILURE_TIMEOUT,
            ),
        )

    def _policy_action(self, state: np.ndarray) -> np.ndarray:
        """Normalize + infer one action from the model (policy failures surface)."""
        if self.normalizer is None:
            raise PolicyInvalidActionError("No normalizer available for policy inference.")
        if self.model is None:
            raise PolicyInvalidActionError("No policy available.")
        device = self._model_device or torch.device("cpu")
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        if self._normalizer_mean is None or self._normalizer_std is None:
            # This fallback keeps the method safe for a custom normalizer
            # implementation while the standard FrozenNormalizer takes the
            # cached path above.
            normalized_state = self.normalizer.normalize(state_tensor)
        else:
            normalized_state = (state_tensor - self._normalizer_mean) / self._normalizer_std
        normalized = normalized_state.unsqueeze(0)
        with torch.inference_mode():
            action_tensor = self.model.get_action(normalized)  # type: ignore[operator]
        action = np.asarray(action_tensor.detach().cpu().numpy()).reshape(-1)
        return self.adapter.validate_action(action, tolerance=self.action_tolerance)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _episode(self, case, outcome: RolloutOutcome) -> tuple[RolloutOutcome, dict[str, Any]]:
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "model": self.model_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "task": self.task,
            "training_seed": self.training_seed,
            "reset_seed": self.bank.seed,
            "episode_index": int(case.index),
            "valid_episode": outcome.valid,
            "steps": int(outcome.steps),
        }
        if self.tag is not None:
            row["tag"] = self.tag
        if outcome.valid:
            row["success"] = outcome.success
            row["timed_out"] = outcome.timed_out
            row["termination_reason"] = outcome.termination_reason
            if not outcome.success:
                row["failure_category"] = outcome.failure_category
                if outcome.exception is not None:
                    row["exception"] = outcome.exception
        else:
            row["termination_reason"] = "infrastructure"
            row["exception"] = outcome.exception or "unknown infrastructure failure"
        if outcome.extra:
            row["extra"] = outcome.extra
        return outcome, row

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def _summarize(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        summaries = summarize_episodes(rows)
        summary = summaries[0] if summaries else {}
        valid = int(summary.get("valid_episodes", 0))
        successes = int(summary.get("successes", 0))
        low, high = wilson_interval(successes, valid)
        policy_failures = int(summary.get("policy_failures", 0))
        results: dict[str, Any] = {
            "eval/action_mse": float("nan"),
            "eval/rollout/success_rate": (successes / valid if valid else float("nan")),
            "eval/rollout/valid_episodes": valid,
            "eval/rollout/successes": successes,
            "eval/rollout/policy_failures": policy_failures,
            "eval/rollout/invalid_attempts": int(summary.get("invalid_attempts", 0)),
            "eval/rollout/wilson_ci95_low": low,
            "eval/rollout/wilson_ci95_high": high,
            "eval/rollout/horizon": self.horizon,
            "eval/rollout/reset_bank": self.bank.bank_id,
            "eval/rollout/reset_seed": self.bank.seed,
            "eval/rollout/task": self.task,
        }
        return results

    def _write_run_summary(self, results: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "model": self.model_name,
            "tag": self.tag,
            "task": self.task,
            "training_seed": self.training_seed,
            "reset_seed": self.bank.seed,
            "reset_bank": self.bank.bank_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "horizon": self.horizon,
            "episodes": len(rows),
            "failure_categories": _failure_breakdown(rows),
            "metrics": {
                key: value for key, value in results.items() if key.startswith("eval/rollout/")
            },
        }
        (self.output_dir / "rollout_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def _failure_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for row in rows:
        if row["valid_episode"] and not row.get("success"):
            category = row.get("failure_category") or "unknown"
            breakdown[category] = breakdown.get(category, 0) + 1
        elif not row["valid_episode"]:
            breakdown["infrastructure"] = breakdown.get("infrastructure", 0) + 1
    return breakdown


# ---------------------------------------------------------------------------
# CLI-facing orchestration
# ---------------------------------------------------------------------------


def state_spec_from_config(cfg: DictConfig) -> StateSpec:
    """Build the declared StateSpec from ``data.state_keys`` (list of dicts)."""
    entries = list(cfg.data.state_keys)
    keys: list[str] = []
    dims: list[int] = []
    for entry in entries:
        keys.append(str(entry["key"]))
        dims.append(int(entry["dim"]))
    spec = StateSpec(keys=tuple(keys), dims=tuple(dims))
    declared = int(cfg.data.get("state_dim", -1))
    if declared != spec.dim:
        raise EnvParityError(
            f"data.state_keys sum to {spec.dim} but data.state_dim declares {declared}."
        )
    data_source = cfg.data.get("source")
    task_name = data_source.get("task_name") if data_source is not None else None
    if task_name is not None:
        from phaseforge.evaluations.envs.task_registry import validate_task_schema

        validate_task_schema(
            str(task_name),
            spec.keys,
            spec.dims,
            int(cfg.data.action_dim),
        )
    return spec


def resolve_pinned_metadata(cfg: DictConfig) -> PinnedEnvMetadata:
    """Recover the pinned env metadata from the cache (or raw HDF5, or dev).

    Priority (fail-closed): processed cache trajectory -> raw HDF5
    ``env_args`` -> documented dev fallback (only when the data source is
    genuinely absent locally; the evaluation machine always hits the
    cache path because training ingests the same dataset).
    """
    from phaseforge.data.ingestion.cache_manager import CacheManager
    from phaseforge.data.paths import processed_cache_root

    hash_val = CacheManager.compute_hash(cfg.data)
    cache_dir = processed_cache_root() / hash_val
    if (cache_dir / "trajectories").is_dir():
        meta = env_metadata_from_cache(cache_dir)
        logger.info("Pinned env metadata recovered from cache %s", hash_val)
        return meta

    raw_dir = cfg.data.source.get("dir")
    hdf5_files = []
    if raw_dir:
        raw_path = Path(str(raw_dir))
        if raw_path.is_dir():
            hdf5_files = sorted(raw_path.glob("*.hdf5"))
    if hdf5_files:
        from phaseforge.evaluations.envs.env_metadata import env_metadata_from_hdf5

        meta = env_metadata_from_hdf5(hdf5_files[0])
        logger.info("Pinned env metadata recovered from raw HDF5 %s", hdf5_files[0])
        return meta

    from phaseforge.evaluations.envs.env_metadata import dev_fallback_metadata

    logger.warning(
        "No processed cache or raw HDF5 found — using the documented dev "
        "fallback env metadata. This is only acceptable for local "
        "self-tests/gates; a real rollout requires the dataset."
    )
    task_name = str(cfg.data.source.get("task_name") or "Lift")
    return dev_fallback_metadata(task_name)


def load_or_generate_bank(cfg: DictConfig, meta: PinnedEnvMetadata) -> ResetBank:
    """Load the frozen bank for the pinned env; generate if configured.

    ``cfg.eval.bank.auto_generate`` (default true) creates the bank on
    first use; the artifact is then frozen and verified by SHA-256 on
    every later load. Regeneration of an existing, hash-verified bank is
    impossible (its ``bank_id`` is content-derived), so existing banks are
    always reused.
    """
    from phaseforge.data.paths import get_data_root

    task = str(cfg.data.source.get("task_name") or meta.env_name)
    seed = int(cfg.eval.bank.get("seed", 2026))
    num_cases = int(cfg.eval.bank.get("num_cases", 50))
    configured_expected_env = cfg.eval.env.get("expected_env_name")
    expected_env_name = (
        str(configured_expected_env) if configured_expected_env is not None else meta.env_name
    )
    versions = verify_environment_parity(
        meta,
        expected_env_name=expected_env_name,
        robosuite_requirement=str(cfg.eval.env.get("robosuite_requirement", "==1.5.1")),
        mujoco_requirement=str(cfg.eval.env.get("mujoco_requirement", "==3.2.7")),
    )
    bank_id = compute_bank_id(
        meta,
        task=task,
        seed=seed,
        num_cases=num_cases,
        robosuite_version=versions["robosuite"],
    )
    directory = bank_dir(get_data_root(), task, bank_id)
    if directory.is_dir():
        bank = ResetBank.load(directory)
        mismatches: list[str] = []
        if bank.bank_id != bank_id:
            mismatches.append(f"bank_id={bank.bank_id!r} (expected {bank_id!r})")
        if bank.task != task:
            mismatches.append(f"task={bank.task!r} (expected {task!r})")
        if bank.seed != seed:
            mismatches.append(f"seed={bank.seed} (expected {seed})")
        if bank.num_cases != num_cases:
            mismatches.append(f"num_cases={bank.num_cases} (expected {num_cases})")
        if bank.env_canonical != meta.canonical_json():
            mismatches.append("env_canonical does not match the pinned dataset metadata")
        if bank.robosuite_version != versions["robosuite"]:
            mismatches.append(
                f"robosuite_version={bank.robosuite_version!r} (expected {versions['robosuite']!r})"
            )
        if mismatches:
            raise EnvParityError(
                f"Reset bank {directory} identity mismatch; refusing to use it: "
                + "; ".join(mismatches)
            )
        logger.info("Loaded frozen reset bank %s (%d cases)", bank_id, bank.num_cases)
        return bank

    auto = bool(cfg.eval.bank.get("auto_generate", True))
    if not auto:
        raise EnvParityError(
            f"Reset bank {bank_id} does not exist under {directory} and "
            "auto_generate is disabled. Generate the bank explicitly "
            "(documented exception: one-time artifact, then frozen)."
        )
    logger.warning(
        "Reset bank %s does not exist — generating it now (one-time "
        "artifact, then frozen and verified on every load).",
        bank_id,
    )
    bank = generate_reset_bank(
        lambda: _adapter_from_config(cfg, meta, seed=seed),
        meta,
        task=task,
        seed=seed,
        num_cases=num_cases,
        max_attempts_per_case=int(cfg.eval.bank.get("max_attempts_per_case", 40)),
        robosuite_version=versions["robosuite"],
        git_commit=_git_commit(),
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    bank.save(directory)
    logger.info("Reset bank %s generated and saved to %s", bank_id, directory)
    return bank


def _git_commit() -> str:
    try:
        from phaseforge.data.ingestion.cache_manager import git_commit

        return git_commit()
    except Exception:  # noqa: BLE001
        return ""


def _adapter_from_config(
    cfg: DictConfig, meta: PinnedEnvMetadata, *, seed: int | None = None
) -> RobosuiteStateAdapter:
    from phaseforge.evaluations.envs.robosuite_adapter import RobosuiteStateAdapter
    from phaseforge.evaluations.envs.task_registry import TaskSpec

    spec = state_spec_from_config(cfg)
    action_contract = cfg.data.action_contract
    task_name = str(cfg.data.source.get("task_name") or meta.env_name)
    # The v1.5.1 PH Transport metadata omits ``horizon``.  In that case use
    # the protocol registry's task-specific default (700 for Transport), not
    # PinnedEnvMetadata's generic legacy fallback of 500.  An explicitly
    # serialized horizon still wins.
    horizon = meta.horizon
    if "horizon" not in meta.env_kwargs:
        horizon = TaskSpec.from_protocol(task_name).horizon
    return RobosuiteStateAdapter(
        meta,
        spec,
        action_dim=int(cfg.data.action_dim),
        action_low=float(action_contract.range[0]),
        action_high=float(action_contract.range[1]),
        horizon=horizon,
        seed=seed,
    )


def run_rollout_evaluation(
    cfg: DictConfig,
    model: torch.nn.Module,
    output_dir: str | Path,
    run_id: str,
    *,
    checkpoint_sha256: str = "",
) -> dict[str, Any]:
    """Full rollout evaluation for one model checkpoint (called by the CLI).

    Order (plan §4.1/§4.4): pinned env metadata -> environment parity gate
    (fail closed) -> frozen reset bank -> normalized rollout over all bank
    cases -> strict-metric episode rows -> per-run summary.
    """
    from phaseforge.data.common.normalizer import FrozenNormalizer
    from phaseforge.data.ingestion.cache_manager import CacheManager
    from phaseforge.data.paths import processed_cache_root

    meta = resolve_pinned_metadata(cfg)
    bank = load_or_generate_bank(cfg, meta)
    adapter = _adapter_from_config(cfg, meta)
    hash_val = CacheManager.compute_hash(cfg.data)
    normalizer = FrozenNormalizer.load(processed_cache_root() / hash_val / "norm_stats.pt")

    model_name = getattr(cfg.models, "name", cfg.models._target_.split(".")[-1])
    training_seed = cfg.project.get("seed")
    if not isinstance(training_seed, int):
        raise EnvParityError(
            "Rollout evaluation requires an integer project seed (shared "
            "across variants for the paired comparisons)."
        )

    evaluator = RolloutEvaluator(
        cfg=cfg,
        policy=None,
        adapter=adapter,
        bank=bank,
        normalizer=normalizer,
        model=model,
        output_dir=output_dir,
        run_id=run_id,
        model_name=model_name,
        training_seed=training_seed,
        task=str(cfg.data.source.get("task_name") or meta.env_name),
        checkpoint_sha256=checkpoint_sha256,
        tag=cfg.project.get("tag"),
        horizon=cfg.eval.episodes.get("horizon"),
        action_tolerance=float(cfg.eval.episodes.get("action_tolerance", 1e-4)),
    )
    try:
        return evaluator.run()
    finally:
        adapter.close()


__all__ = [
    "RolloutEvaluator",
    "RolloutOutcome",
    "RolloutRunInvalid",
    "run_rollout_evaluation",
    "resolve_pinned_metadata",
    "load_or_generate_bank",
    "state_spec_from_config",
]
