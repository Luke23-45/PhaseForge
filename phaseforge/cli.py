"""Command line interfaces for training and evaluation.

Entry points:
    phaseforge-train: Runs the training loop (Stage 1 or Stage 2).
    phaseforge-eval: Runs the evaluation loop.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from phaseforge.outputs_writer.ledger import LedgerRow, RunLedger
from phaseforge.outputs_writer.metadata import collect_environment
from phaseforge.outputs_writer.provenance import (
    copy_cache_provenance,
    write_artifact_manifest,
)
from phaseforge.outputs_writer.results import append_result_row
from phaseforge.outputs_writer.schema import OPTIONAL_METRIC_FIELDS
from phaseforge.outputs_writer.training_summary import (
    append_training_summary_row,
    write_reconciliation_record,
)
from phaseforge.outputs_writer.writer import RunWriter, parse_run_dir
from phaseforge.trains.callbacks.checkpointing import CheckpointCallback
from phaseforge.trains.callbacks.early_stopping import EarlyStoppingCallback
from phaseforge.trains.callbacks.metric_tracker import MetricTrackerCallback
from phaseforge.trains.callbacks.persistence import MetricPersistenceCallback
from phaseforge.trains.callbacks.wandb_logger import WandbLoggerCallback
from phaseforge.utils.config import (
    checkpoint_source_info,
    config_hash,
    find_latest_checkpoint,
    get_eval_output_dir,
    get_output_dir,
    git_info,
    output_base_dir,
    resolve_checkpoint_source,
    write_run_meta,
)
from phaseforge.utils.registry import build_data_pipeline, build_model, build_trainer
from phaseforge.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _apply_log_level(cfg: DictConfig) -> None:
    """Filter the run's log stream to the configured verbosity.

    Hydra configures the root logger at INFO on startup, which floods cloud
    consoles with per-epoch routing diagnostics, checkpoint chatter, and
    third-party INFO lines (e.g. numexpr). ``project.log_level`` (default
    ``WARNING``) raises the root level so only warnings and errors pass;
    child loggers without an explicit level (all of ``phaseforge.*`` and
    most third-party loggers) inherit the filter. Set
    ``project.log_level=INFO`` (or ``DEBUG``) for verbose local runs.
    """
    level_name = str(cfg.project.get("log_level", "WARNING")).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        logger.warning("Unknown log_level %r — falling back to WARNING.", level_name)
        level = logging.WARNING
    logging.getLogger().setLevel(level)
    logger.info(
        "Log level: %s (override with project.log_level=INFO or DEBUG).",
        logging.getLevelName(level),
    )


def _resolve_device(cfg: DictConfig) -> torch.device:
    """Resolve the compute device with a graceful CPU fallback.

    Offline evaluation is reproducible on CPU; requiring a hard crash when
    ``project.device=cuda`` is unavailable makes local development brittle.
    A warning is logged when falling back.
    """
    requested = str(cfg.project.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "Device '%s' requested but CUDA is unavailable. Falling back to CPU.",
            requested,
        )
        return torch.device("cpu")
    return torch.device(requested)


def _init_run_bookkeeping(
    cfg: DictConfig, output_dir: Path, *, kind: str
) -> tuple[RunWriter, RunLedger, str]:
    """Create the RunWriter + pending ledger row for a run directory.

    Called at the very start of a run (before the data pipeline or model
    construction) so a killed cloud session still leaves a trace in
    ``outputs/_ledger/``.

    Returns ``(run_writer, ledger, run_id)``. The environment fingerprint
    is written immediately; the ledger row is ``pending`` until the
    caller flips it via the run-specific finalizer
    (:func:`_finalize_training_run` / :func:`_finalize_eval_run` /
    :func:`_mark_run_failed`).
    """
    from phaseforge.data.ingestion.cache_manager import CacheManager

    outputs_base = output_base_dir(cfg)
    ledger = RunLedger(outputs_base / "_ledger")
    run_writer = RunWriter(output_dir)
    timestamp, tag, run_id = parse_run_dir(output_dir.name)
    run_writer.write_environment(
        collect_environment(
            data_config_hash=CacheManager.compute_hash(cfg.data),
            config_hash=config_hash(cfg),
            extra={"kind": kind, "device": str(cfg.project.device), "tag": tag},
        )
    )
    ledger.append(
        LedgerRow(
            run_id=run_id,
            kind=kind,
            timestamp=timestamp,
            model=getattr(cfg.models, "name", cfg.models._target_.split(".")[-1]),
            stage=cfg.train.get("stage", 1),
            seed=cfg.project.get("seed"),
            config_hash=config_hash(cfg),
            git_sha=git_info()["commit"],
            status="pending",
            path=str(output_dir),
            extra={"tag": tag} if tag else {},
        )
    )
    return run_writer, ledger, run_id


def _mark_run_failed(
    run_writer: RunWriter,
    ledger: RunLedger,
    run_id: str,
    exc: BaseException,
) -> None:
    """Mark a run failed (recording the exception); re-raise by the caller."""
    try:
        run_writer.mark_failed(exc)
        ledger.update_status(run_id, "failed")
    except Exception:
        logger.exception(
            "Failed to record the failed run (%s) in outputs bookkeeping.", run_id
        )


def _append_eval_result_row(
    cfg: DictConfig,
    model: torch.nn.Module,
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Append one schema-validated row to the global results ledger.

    The row's ``stage`` is the stage restored from the evaluated checkpoint
    (not the default ``train`` group); ``ckpt_path`` records the exact
    artifact evaluated so the paper tables can be traced to a checkpoint.
    """
    _timestamp, _tag, run_id = parse_run_dir(output_dir.name)
    git = git_info()
    row: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": _timestamp,
        "model": getattr(cfg.models, "name", cfg.models._target_.split(".")[-1]),
        "stage": int(getattr(model, "stage", cfg.train.get("stage", 1))),
        "seed": cfg.project.get("seed"),
        "git_sha": git["commit"],
        "config_hash": config_hash(cfg),
        "device": str(cfg.project.get("device")),
        "ckpt_path": str(cfg.train.get("stage1_ckpt_path") or ""),
        "action_mse": float(results["eval/action_mse"]),
    }
    for metric in OPTIONAL_METRIC_FIELDS:
        key = f"eval/{metric}"
        if key in results:
            row[metric] = float(results[key])
    append_result_row(output_base_dir(cfg) / "_results", row)


def _unused_stage1_head_prefixes(model: torch.nn.Module) -> tuple[str, ...]:
    """Stage 1 head prefixes the target model does not use at all.

    The Stage 1 checkpoint of a phase-supervised cell (``phaseforge``)
    contains ``phase_head`` weights. A Stage 2 cell without a phase head
    (e.g. ``phase_pretrain_random_router``) legitimately drops them; a cell
    that routes by the phase head (``phaseforge``, ``teacher_forced``) must
    always load them exactly. Heads the target model owns are therefore
    never droppable.
    """
    return tuple(
        prefix
        for prefix in ("phase_head",)
        if not any(key.startswith(prefix) for key in model.state_dict())
    )


def _load_state_dict_checked(
    model: torch.nn.Module,
    state_dict: dict,
    context: str,
    expected_unexpected_prefixes: tuple[str, ...] = (),
) -> None:
    """Load a checkpoint state dict and hard-fail on any unexpected mismatch.

    ``strict=False`` exists so a Stage 1 checkpoint can be loaded into a
    Stage 2 model before bootstrapping (the MoE block is legitimately
    absent). Everything else is a config mismatch: silently skipping it
    leaves the policy on random weights (which can produce meaningless
    rollout results with no error), so mismatched weights stop the run instead.

    Args:
        model: The model to load into.
        state_dict: The checkpoint's ``model_state_dict``.
        context: Human-readable label for error messages.
        expected_unexpected_prefixes: Parameter-name prefixes that are
            legitimately different between the checkpoint and target model.
            This covers both missing target keys and unexpected checkpoint
            keys (e.g. ``moe_layer`` when loading a BC Stage 1 checkpoint
            into a Stage 2 MoE model).
    """
    result = model.load_state_dict(state_dict, strict=False)
    expected_missing = [
        key
        for key in result.missing_keys
        if any(key.startswith(prefix) for prefix in expected_unexpected_prefixes)
    ]
    missing = [key for key in result.missing_keys if key not in expected_missing]
    unexpected = [
        key
        for key in result.unexpected_keys
        if not any(key.startswith(prefix) for prefix in expected_unexpected_prefixes)
    ]

    expected_count = len(expected_missing) + (
        len(result.unexpected_keys) - len(unexpected)
    )
    if expected_count and not missing and not unexpected:
        logger.info(
            "%s: %d key(s) differ under prefix(es) %s — expected, "
            "skipped.",
            context, expected_count, ", ".join(expected_unexpected_prefixes) or "-",
        )
        return
    if not missing and not unexpected:
        return

    raise RuntimeError(
        f"{context}: checkpoint/model mismatch — {len(missing)} "
        f"missing, {len(unexpected)} unexpected key(s) outside the allowed "
        f"prefix(es) {expected_unexpected_prefixes or '-'}. "
        f"Missing (first 8): {missing[:8] or '-'}. "
        f"Unexpected (first 8): {unexpected[:8] or '-'}. "
        "Refusing to continue with randomly initialized weights — fix the "
        "eval/training model config to match the checkpoint's training config."
    )


@hydra.main(version_base="1.3", config_path="config", config_name="main")
def train(cfg: DictConfig) -> None:
    """Main training entry point."""
    # 1. Setup
    _apply_log_level(cfg)
    set_seed(cfg.project.seed)
    # Resolve once and write the effective device back into the config so
    # bootstrap, trainer construction, and run metadata all use the same
    # CPU fallback when CUDA is unavailable.
    effective_device = _resolve_device(cfg)
    cfg.project.device = str(effective_device)

    # Per-model override: baselines without Stage 1 pretraining (scratch_moe,
    # oracle_moe) must train their encoder in Stage 2. Apply before the
    # resolved config / run metadata are written so records are accurate.
    model_freeze = cfg.models.get("freeze_encoder", None)
    if model_freeze is not None:
        cfg.train.freeze_encoder = bool(model_freeze)

    output_dir = get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config and lightweight run metadata. run_meta.json
    # records the effective data-config hash (final specification §5.5),
    # mirroring the eval path and environment.json.
    from phaseforge.data.ingestion.cache_manager import CacheManager

    with open(output_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    write_run_meta(
        output_dir,
        cfg,
        kind="train",
        data_config_hash=CacheManager.compute_hash(cfg.data),
    )

    logger.info(f"Output directory: {output_dir}")

    # 2. Run lifecycle bookkeeping BEFORE any work so a killed cloud session
    #    still leaves a pending ledger row + environment fingerprint.
    run_writer, ledger, run_id = _init_run_bookkeeping(cfg, output_dir, kind="train")
    try:
        _train_body(cfg, output_dir, run_id)
    except BaseException as exc:
        _mark_run_failed(run_writer, ledger, run_id, exc)
        raise
    _finalize_training_run(cfg, output_dir, run_writer, ledger, run_id)


def _finalize_training_run(
    cfg: DictConfig,
    output_dir: Path,
    run_writer: RunWriter,
    ledger: RunLedger,
    run_id: str,
) -> None:
    """Complete a successful training run in the provenance-defined order.

    Order (final specification §5.3 / §9.3 / §9.4):

    1. Append the schema-validated ``training_summary.jsonl`` row — or, on
       failure, write a reconciliation record so the run-local summary can
       rebuild the ledger. The run is never marked complete before this.
    2. Copy the cache data-provenance into ``metadata/data_provenance.json``.
    3. Mark the run completed (writes ``timings.json`` + the sibling marker).
    4. Hash every paper input into ``metadata/artifact_manifest.json`` —
       written after step 3 so ``timings.json`` exists to be hashed.
    5. Flip the ledger row to ``completed``.

    Steps 2–5 are best effort in the same spirit as the eval result row:
    a bookkeeping failure must not flip an already-successful training run
    to failed, but every failure is logged loudly and the run-local
    artifacts remain authoritative for recovery.
    """
    try:
        _append_training_summary_row(cfg, output_dir)
    except Exception as exc:
        logger.exception(
            "Failed to append the training summary row for run %s — wrote a "
            "reconciliation record; run-local summary.json is authoritative.",
            run_id,
        )
        try:
            write_reconciliation_record(output_dir, exc)
        except Exception:
            logger.exception(
                "Failed to write the reconciliation record for run %s.", run_id
            )
    try:
        _copy_data_provenance(cfg, output_dir)
    except Exception:
        logger.exception(
            "Failed to copy the cache data provenance into %s.", output_dir.name
        )
    try:
        run_writer.mark_completed()
    except Exception:
        logger.exception("Failed to mark run %s completed.", run_id)
    try:
        write_artifact_manifest(
            output_dir,
            {
                "resolved_config.yaml": "resolved_config.yaml",
                "run_meta.json": "run_meta.json",
                "metadata/environment.json": "metadata/environment.json",
                "metadata/data_provenance.json": "metadata/data_provenance.json",
                "timings.json": "timings.json",
                "metrics/training_curves.jsonl": "metrics/training_curves.jsonl",
                "metrics/summary.json": "metrics/summary.json",
                "checkpoints/checkpoint_best.pt": "checkpoints/checkpoint_best.pt",
            },
        )
    except Exception:
        logger.exception("Failed to write the artifact manifest for run %s.", run_id)
    try:
        ledger.update_status(run_id, "completed")
    except Exception:
        logger.exception(
            "Failed to update the ledger for completed run %s.", run_id
        )


def _append_training_summary_row(cfg: DictConfig, output_dir: Path) -> None:
    """Append the run-local summary to the global training ledger.

    The run-local ``metrics/summary.json`` (written by the persistence
    callback at ``on_train_end``) is authoritative. On any failure a
    reconciliation record is written next to it so
    ``reconcile_training_ledger`` can rebuild the ledger — provenance is
    never silently lost.
    """
    summary_path = output_dir / "metrics" / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"No run-local summary at {summary_path} — the persistence "
            "callback did not write one."
        )
    import json as _json

    summary = _json.loads(summary_path.read_text(encoding="utf-8"))
    base_dir = output_base_dir(cfg)
    try:
        summary["run_dir"] = output_dir.relative_to(base_dir).as_posix()
    except ValueError:
        summary["run_dir"] = str(output_dir)
    append_training_summary_row(base_dir / "_results", summary)


def _copy_data_provenance(cfg: DictConfig, output_dir: Path) -> None:
    """Copy the cache manifest's provenance block into the run directory.

    Uses the effective cache hash recorded in ``environment.json`` when
    available (the pipeline may have fallen back to a different cache),
    else the freshly computed data-config hash.
    """
    from phaseforge.data.ingestion.cache_manager import CacheManager
    from phaseforge.data.paths import processed_cache_root

    env_path = output_dir / "metadata" / "environment.json"
    config_hash_val = CacheManager.compute_hash(cfg.data)
    if env_path.is_file():
        try:
            import json as _json

            env = _json.loads(env_path.read_text(encoding="utf-8"))
            recorded = env.get("data_config_hash")
            if recorded:
                config_hash_val = str(recorded)
        except (OSError, _json.JSONDecodeError):
            logger.warning(
                "Could not read environment.json to resolve the effective "
                "cache hash — falling back to the computed data-config hash.",
                exc_info=True,
            )
    copy_cache_provenance(
        output_dir, processed_cache_root(), config_hash_val
    )


def _train_body(cfg: DictConfig, output_dir: Path, run_id: str) -> None:
    """The training work itself, wrapped in run lifecycle bookkeeping."""
    # 2. Init W&B (lazy import: cli.py stays importable without wandb installed)
    try:
        import wandb
    except ImportError:
        wandb = None
    if wandb is not None and cfg.project.wandb.mode != "disabled":
        wandb.init(
            project=cfg.project.wandb.project,
            entity=cfg.project.wandb.entity,
            mode=cfg.project.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=str(output_dir),
        )

    # 3. Data Pipeline
    logger.info("Initializing Data Pipeline...")
    pipeline = build_data_pipeline(cfg)
    dataloaders = pipeline.run()
    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("val")

    if train_loader is None:
        raise RuntimeError("No training data found. Check split ratios and cache.")

    # 4. Model
    logger.info("Initializing Model...")
    model = build_model(cfg)

    stage = cfg.train.get("stage", 1)

    if stage == 2:
        ckpt_path = cfg.train.get("stage1_ckpt_path")

        if hasattr(model, "bootstrap_moe"):
            # Models with bootstrapping (PhaseBootstrappedMoE, WarmStartMoE)
            # need a Stage 1 checkpoint to initialise encoder + action_head.
            if not ckpt_path:
                model_name = getattr(cfg.models, "name", cfg.models._target_.split(".")[-1])
                source_model = resolve_checkpoint_source(model_name)
                auto_ckpt = find_latest_checkpoint(
                    source_model, stage=1, base=cfg.project.output_dir,
                    resolve_alias=False, seed=cfg.project.get("seed"),
                    require_seed=cfg.project.get("seed") is not None,
                )
                if auto_ckpt is not None:
                    ckpt_path = str(auto_ckpt)
                    logger.info(
                        f"Auto-detected Stage 1 checkpoint (from '{source_model}'): {ckpt_path}"
                    )
                else:
                    raise ValueError(
                        f"{type(model).__name__} requires a Stage 1 checkpoint. "
                        f"Set train.stage1_ckpt_path or ensure "
                        f"outputs/{source_model}/stage1/ has one."
                    )

            # Record the RESOLVED source so run metadata, environment.json,
            # the summary's source_stage1, and the artifact manifest all
            # reflect the exact checkpoint actually loaded (auto-detection
            # can select a different artifact after later runs are added).
            cfg.train.stage1_ckpt_path = ckpt_path

            logger.info(f"Loading Stage 1 checkpoint from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            _load_state_dict_checked(
                model,
                ckpt["model_state_dict"],
                "Stage 1 -> Stage 2 bootstrap load",
                expected_unexpected_prefixes=(
                    "moe_layer",
                ) + _unused_stage1_head_prefixes(model),
            )

            model.bootstrap_moe(dataloader=train_loader, device=cfg.project.get("device", "cuda"))
        else:
            # Models without bootstrapping (ScratchMoE, OraclePhaseMoE)
            # train from scratch — no checkpoint needed.
            logger.info(f"{type(model).__name__}: No bootstrapping. Training from scratch.")

    # 5. Trainer
    logger.info(f"Initializing Stage {stage} Trainer...")
    trainer = build_trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # 6. Callbacks
    trainer.add_callback(MetricTrackerCallback())

    # Persist the per-epoch curves + final summary. Stage 2 runs record the
    # exact Stage 1 source artifact (final specification §4.1) so a later
    # run cannot silently change the auto-detected source.
    source_stage1 = None
    if stage == 2 and cfg.train.get("stage1_ckpt_path"):
        source_stage1 = checkpoint_source_info(
            cfg.train.stage1_ckpt_path, base=cfg.project.output_dir
        )
    from phaseforge.data.ingestion.cache_manager import CacheManager

    trainer.add_callback(MetricPersistenceCallback(
        run_dir=output_dir,
        run_id=run_id,
        data_config_hash=CacheManager.compute_hash(cfg.data),
        source_stage1=source_stage1,
    ))

    if hasattr(cfg.train, "early_stopping") or "early_stopping" in cfg.train:
        if cfg.train.early_stopping.get("enabled", True):
            trainer.add_callback(EarlyStoppingCallback(
                monitor=cfg.train.early_stopping.monitor,
                mode=cfg.train.early_stopping.mode,
                patience=cfg.train.early_stopping.patience,
                min_delta=cfg.train.early_stopping.min_delta,
            ))

    # Save after metric tracking and early stopping have processed the epoch,
    # so checkpoint resume restores their post-epoch state.
    trainer.add_callback(CheckpointCallback(
        output_dir=output_dir / "checkpoints",
        every_n_epochs=cfg.train.checkpoint.every_n_epochs,
        monitor=cfg.train.checkpoint.monitor,
        mode=cfg.train.checkpoint.mode,
        save_top_k=cfg.train.checkpoint.save_top_k,
    ))

    if cfg.project.wandb.mode != "disabled":
        trainer.add_callback(WandbLoggerCallback())

    # 7. Resume (optional): restore model/optimizer/scheduler/RNG/callback
    # state from an interrupted run before training continues.
    resume_path = cfg.train.get("resume_from")
    if resume_path:
        resume_ckpt = Path(resume_path)
        if not resume_ckpt.is_file():
            raise FileNotFoundError(
                f"train.resume_from={resume_path} is not a file. "
                "Set it to a checkpoint saved by CheckpointCallback or remove it."
            )
        logger.info(f"Resuming training from {resume_ckpt}")
        trainer.resume(resume_ckpt)

    # 8. Go!
    trainer.fit()

    if wandb is not None and wandb.run is not None:
        wandb.finish()


def build_eval_model(cfg: DictConfig) -> torch.nn.Module:
    """Build the configured model and load the evaluation checkpoint.

    Used by both the CLI entry point and the parallel rollout workers so
    that eval-time model construction (architecture, checkpoint, stage
    restore) is identical in every process.

    Returns:
        The model in ``eval()`` mode, moved to CPU (caller decides device).
    """
    model = build_model(cfg)

    ckpt_path = cfg.train.get("stage1_ckpt_path")
    if ckpt_path:
        logger.info(f"Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _load_state_dict_checked(
            model, ckpt["model_state_dict"], "Evaluation checkpoint load"
        )
        # Restore the stage attribute — it is a plain Python int, NOT in state_dict(),
        # so load_state_dict() leaves it at the __init__ default (1).
        if hasattr(model, "stage") and "stage" in ckpt:
            model.stage = ckpt["stage"]
    else:
        logger.warning(
            "No checkpoint provided (train.stage1_ckpt_path). "
            "Using randomly initialized model."
        )
    return model


@hydra.main(version_base="1.3", config_path="config", config_name="main")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model with the currently supported offline metrics."""
    _apply_log_level(cfg)
    set_seed(cfg.project.seed)
    # Resolve the effective device and write it back BEFORE any run artifact
    # is written, so resolved_config.yaml, environment.json, the ledger row,
    # run_meta.json, and the results row all record the same device (mirrors
    # the train() ordering).
    effective_device = _resolve_device(cfg)
    cfg.project.device = str(effective_device)
    output_dir = get_eval_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    logger.info(f"Evaluation output directory: {output_dir}")

    # 2. Run lifecycle bookkeeping BEFORE the pipeline so a killed session
    #    still leaves a pending ledger row + environment fingerprint.
    run_writer, ledger, run_id = _init_run_bookkeeping(cfg, output_dir, kind="eval")
    try:
        _eval_body(cfg, output_dir)
    except BaseException as exc:
        _mark_run_failed(run_writer, ledger, run_id, exc)
        raise
    _finalize_eval_run(cfg, output_dir, run_writer, ledger, run_id)


def _finalize_eval_run(
    cfg: DictConfig,
    output_dir: Path,
    run_writer: RunWriter,
    ledger: RunLedger,
    run_id: str,
) -> None:
    """Complete a successful eval run: mark done, then hash its inputs."""
    try:
        run_writer.mark_completed()
    except Exception:
        logger.exception("Failed to mark eval run %s completed.", run_id)
    try:
        write_artifact_manifest(
            output_dir,
            {
                "resolved_config.yaml": "resolved_config.yaml",
                "run_meta.json": "run_meta.json",
                "metadata/environment.json": "metadata/environment.json",
                "timings.json": "timings.json",
                "eval_results.json": "eval_results.json",
            },
        )
    except Exception:
        logger.exception("Failed to write the artifact manifest for eval run %s.", run_id)
    try:
        ledger.update_status(run_id, "completed")
    except Exception:
        logger.exception(
            "Failed to update the ledger for completed eval run %s.", run_id
        )


def _eval_body(cfg: DictConfig, output_dir: Path) -> None:
    """The evaluation work itself, wrapped in run lifecycle bookkeeping."""
    eval_mode = cfg.eval.get("mode", "offline")
    logger.info(f"Evaluation mode: {eval_mode}")

    # 1. Data Pipeline — only needed for offline metrics. Rollout evaluation
    #    reads the training-frozen normalizer straight from the processed
    #    cache and must NOT load the full dataset into RAM.
    val_loader = None
    if eval_mode != "rollout":
        logger.info("Initializing Data Pipeline...")
        pipeline = build_data_pipeline(cfg)
        dataloaders = pipeline.run()
        val_loader = dataloaders.get("val") or dataloaders.get("test")
        if val_loader is None:
            raise RuntimeError("No validation/test data found for evaluation.")

    # 2. Model
    logger.info("Initializing Model...")
    model = build_eval_model(cfg)
    # Device is resolved + written back by the evaluate() wrapper BEFORE any
    # run artifact is written; metadata reflects the artifact actually
    # evaluated: the stage restored from the checkpoint (an eval run's
    # `train` group is stage1 by default).
    from phaseforge.data.ingestion.cache_manager import CacheManager

    write_run_meta(
        output_dir,
        cfg,
        stage=getattr(model, "stage", None),
        kind="eval",
        data_config_hash=CacheManager.compute_hash(cfg.data),
    )
    model.to(torch.device(cfg.project.device))

    # 3. Run the selected evaluator
    if eval_mode == "rollout":
        raise ValueError(
            "Rollout evaluation is not implemented for the current protocol. "
            "The robomimic/robosuite protocol requires a separate simulator "
            "adapter, reset distribution, success predicate, and paired test "
            "episode runner. Use `eval.mode: offline` until that evaluator is "
            "implemented and validated."
        )
    else:
        logger.info("Running offline evaluation on validation data…")
        from phaseforge.evaluations.runners.offline_evaluator import OfflineEvaluator

        evaluator = OfflineEvaluator(cfg=cfg, model=model, dataloader=val_loader)

    results = evaluator.run()

    # 4. Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Evaluation complete:")
    for key, val in results.items():
        if isinstance(val, float):
            logger.info(f"  {key}: {val:.6f}")
        elif isinstance(val, dict):
            logger.info(f"  {key}: <dict with {len(val)} entries>")
        else:
            logger.info(f"  {key}: {val}")
    logger.info(f"Results saved to {results_path}")

    # 5. Append the schema-validated row to the global results ledger
    #    (outputs/_results/results.jsonl) — the aggregation source. Best
    #    effort: the evaluation already succeeded (eval_results.json is
    #    saved), so a bookkeeping failure must NOT flip the run to failed.
    try:
        _append_eval_result_row(cfg, model, results, output_dir)
    except Exception:
        logger.exception(
            "Failed to append the eval result row for run %s — results "
            "remain in %s. Fix the ledger and re-run summarize_eval.py.",
            output_dir.name,
            results_path,
        )
