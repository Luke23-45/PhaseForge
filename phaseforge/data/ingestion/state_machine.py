"""Data ingestion state machine.

This is the single entry point for all data loading. The trainer calls
``DataPipelineStateMachine(cfg).run()`` and receives a dict of DataLoaders.

States
------
CHECK_PERSISTENT_CACHE → (hit) READY
                       → (miss) VALIDATE_SOURCE → INGEST_AND_STRIP
                                               → NORMALIZE_AND_SAVE → READY

When the raw source is missing and ``data.source.auto_download=true``,
VALIDATE_SOURCE routes to PROVISION_SOURCE, which downloads the configured
HuggingFace artifact (SHA-256-verified) before INGEST_AND_STRIP.

Design notes (bugs fixed here, each proven by simulation)
----------------------------------------------------------
- Bug 1: No more fictional box.com download. The FSM consumes pre-downloaded
  data from the env-var-aware data root (paths.py). VALIDATE_SOURCE checks
  the configured ``data.source`` directory before ingesting; downloading
  from the mirror happens only in the explicit PROVISION_SOURCE state when
  ``data.source.auto_download=true`` (off by default, so training runs stay
  fail-closed).
- Bug 4: The processed cache lives under {data_root}/processed/cache, NOT under
  the per-run outputs/ directory, so the config-hash cache is reused across runs.
- Bug 5: Splits are done at trajectory or task level according to the
  protocol, never by timestep; the primary single-task config uses trajectory
  splits so validation is available without trajectory leakage.
- Bug 2: task_id is deterministic (sorted-name -> int); the ingester builds it.

Benchmark contract (docs/plan/final_evaluation_plan.md, Gate 0)
----------------------------------------------------------------
The dataset-specific conversion (robomimic HDF5 -> trajectory dicts) lives in
a pluggable ``data.ingester`` instantiated by :meth:`_ingest_source`. The
former benchmark-specific ingestion is archived under ``legacy/`` and must
not be resurrected here; the robomimic adapter defines its own observation
schema, action contract, splits and phase labels.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from phaseforge.data.common.collator import PhaseAwareCollator
from phaseforge.data.common.dataset import StateOnlyDataset
from phaseforge.data.common.normalizer import RunningStatNormalizer
from phaseforge.data.ingestion.cache_manager import (
    CacheManager,
    git_commit,
    sha256_file,
)
from phaseforge.data.ingestion.hf_downloader import download_hf_file
from phaseforge.data.ingestion.states import PipelineState
from phaseforge.data.paths import processed_cache_root

logger = logging.getLogger(__name__)

DEFAULT_VAL_RATIO = 0.1
DEFAULT_SPLIT_SEED = 42


class PipelineError(RuntimeError):
    """Raised when the pipeline reaches the ERROR state."""


class DataPipelineStateMachine:
    """Autonomous data pipeline implemented as a finite state machine.

    Args:
        cfg: Root Hydra config. The pipeline uses ``cfg.data`` (including
             the generic ``data.source`` and ``data.ingester`` blocks).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.data_cfg = cfg.data

        # Bug 3 (latent): num_phases is repeated across data + model configs
        # with no validation. The phase-count ablation would silently break
        # cross_entropy / scatter_add / bincount if they diverged. Guard here.
        self._check_num_phases_consistency()

        # Bug 4: cache under the shared data root, NOT under outputs/.
        cache_root = processed_cache_root()
        self.cache_manager = CacheManager(cache_root)
        self.config_hash = CacheManager.compute_hash(self.data_cfg)

        self._state = PipelineState.CHECK_PERSISTENT_CACHE
        self._error_msg: str = ""

        # Resolved during processing
        self._task_index: dict[str, int] = {}
        self._trajectories: list[dict[str, Any]] = []
        self._norm_stats: dict[str, torch.Tensor] = {}
        self._splits: dict[str, list[int]] = {}
        self._raw_dir: Path | None = None

    # ------------------------------------------------------------------
    # Config consistency
    # ------------------------------------------------------------------

    def _check_num_phases_consistency(self) -> None:
        """Guard against the latent num_phases mismatch bug.

        The integer num_phases is repeated in:
        - data.phase_labeler.num_phases   (label generation, when present)
        - models.phase_head.num_phases    (classifier width)
        - models.router.num_experts       (router width)
        - models.num_phases               (oracle model)

        Nothing validated they match. If they diverge, the labeler produces
        values the classifier/scatter/bincount cannot index. This guard makes
        the inconsistency a loud, early failure instead of a silent corruption.

        Only checks model fields that exist in the current config (so a
        config that omits the oracle's top-level num_phases doesn't trip it),
        and is skipped entirely when the data config carries no phase-labeler
        block.

        The router width is exempted for PhaseBootstrappedMoE prototype /
        unsupervised / random inits (``centroid``, ``phase_centroid``,
        ``spherical_centroid``, ``spherical_kmeans``, ``kmeans``, ``random``):
        the hierarchical prototype construction supports E != P
        (super-prototypes E<P, intra-phase scaling E>P) and unsupervised
        clustering never indexes phase labels, so the Wave-2 expert-scaling
        sweep (K=3/12 vs P=6) is legal. The 1:1 phase->expert mapping stays
        mandatory for the phase-head copy init (E rows copied from P rows)
        and for models without a ``router_init`` block (teacher-forced hard
        dispatch, oracle).
        """
        labeler_cfg = self.data_cfg.get("phase_labeler")
        if labeler_cfg is None or labeler_cfg.get("num_phases") is None:
            return

        data_phases = int(labeler_cfg.num_phases)

        models_cfg = self.cfg.get("models")
        if models_cfg is None:
            return  # model config not part of this run (e.g. eval-only)

        router_init_type: str | None = None
        router_init_cfg = models_cfg.get("router_init")
        if router_init_cfg is not None and router_init_cfg.get("type") is not None:
            router_init_type = str(router_init_cfg.type).lower()

        candidates: list[tuple[str, int]] = []
        phase_head = models_cfg.get("phase_head")
        if phase_head is not None and phase_head.get("num_phases") is not None:
            candidates.append(("models.phase_head.num_phases", int(phase_head.num_phases)))

        router = models_cfg.get("router")
        e_ne_p_capable = router_init_type in (
            "centroid",
            "phase_centroid",
            "spherical_centroid",
            "spherical_kmeans",
            "kmeans",
            "random",
        )
        if router is not None and router.get("num_experts") is not None and not e_ne_p_capable:
            candidates.append(("models.router.num_experts", int(router.num_experts)))

        # Oracle model carries a top-level num_phases; other models may not.
        if models_cfg.get("num_phases") is not None:
            candidates.append(("models.num_phases", int(models_cfg.num_phases)))

        mismatches = [(name, val) for name, val in candidates if val != data_phases]
        if mismatches:
            details = ", ".join(f"{n}={v}" for n, v in mismatches)
            raise PipelineError(
                f"num_phases inconsistency: data.phase_labeler.num_phases="
                f"{data_phases} but {details}. All must match, otherwise the "
                "phase labels cannot be indexed by the classifier/router/scatter."
            )

    def _model_uses_phase_labels(self) -> bool:
        """True when the selected model consumes phase labels in training.

        Models with a ``phase_head`` (phase cross-entropy supervision) or a
        top-level ``num_phases`` (privileged oracle routing / centroid
        bootstrap) need every phase to be populated; degenerate labels would
        silently corrupt them. MoE rows that declare only a
        ``router.num_experts`` (scratch/warm-start MoE) train without labels
        and must not block the BC pilot on degenerate labels.
        """
        models_cfg = self.cfg.get("models")
        if models_cfg is None:
            return False
        phase_head = models_cfg.get("phase_head")
        if phase_head is not None and phase_head.get("num_phases") is not None:
            return True
        return models_cfg.get("num_phases") is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, DataLoader]:
        """Execute the FSM to the READY terminal state.

        Returns:
            Dict of ``{"train": DataLoader, "val": DataLoader}``.
            Splits that have 0 samples return None in their slot.
        """
        logger.info(f"Pipeline starting. Config hash: {self.config_hash}")

        while self._state not in (PipelineState.READY, PipelineState.ERROR):
            self._step()

        if self._state == PipelineState.ERROR:
            raise PipelineError(self._error_msg)

        return self._build_dataloaders()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _step(self) -> None:
        try:
            if self._state == PipelineState.CHECK_PERSISTENT_CACHE:
                self._check_cache()
            elif self._state == PipelineState.VALIDATE_SOURCE:
                self._validate_source()
            elif self._state == PipelineState.PROVISION_SOURCE:
                self._provision_source()
            elif self._state == PipelineState.INGEST_AND_STRIP:
                self._ingest_source()
            elif self._state == PipelineState.NORMALIZE_AND_SAVE:
                self._normalize_and_save()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline error")
            self._error_msg = str(exc)
            self._state = PipelineState.ERROR

    def _check_cache(self) -> None:
        logger.info("CHECK_PERSISTENT_CACHE: looking for cached data…")
        enforce_strict = self.data_cfg.get("enforce_strict_cache", True)
        found_hash = self.cache_manager.find_cache(self.config_hash, enforce_strict)

        if found_hash:
            logger.info(f"Cache hit (hash: {found_hash}). Loading from disk.")
            (
                self._trajectories,
                self._norm_stats,
                self._splits,
                self._task_index,
            ) = self.cache_manager.load(found_hash)

            # Use the loaded hash so it doesn't mismatch later if we need it
            self.config_hash = found_hash
            self._state = PipelineState.READY
        else:
            logger.info("Cache miss. Proceeding to validate source.")
            self._state = PipelineState.VALIDATE_SOURCE

    def _resolve_raw_dir(self) -> Path:
        """Resolve the configured raw dataset directory (``data.source.dir``)."""
        source = self.data_cfg.get("source")
        if source is None or not source.get("dir"):
            raise PipelineError(
                "data.source.dir is not configured. The robomimic protocol "
                "(docs/plan/final_evaluation_plan.md, Gate 0) requires a "
                "source block with the raw dataset directory before the "
                "pipeline can ingest. Legacy benchmark-specific data configs "
                "are archived under legacy/ and are not part of the protocol."
            )
        return Path(str(source["dir"]))

    def _auto_download_enabled(self) -> bool:
        """True when the FSM may provision missing raw data from the mirror."""
        source = self.data_cfg.get("source")
        if source is None:
            return False
        return bool(source.get("auto_download", False))

    def _validate_source(self) -> None:
        """Bug 1 fix: consume pre-downloaded data and verify it exists."""
        logger.info("VALIDATE_SOURCE: checking pre-downloaded raw data…")
        raw_suite_dir = self._resolve_raw_dir()

        if not raw_suite_dir.exists():
            if self._auto_download_enabled():
                logger.info(
                    "  Raw source directory missing — auto-download enabled, "
                    "provisioning from the configured HuggingFace mirror."
                )
                self._raw_dir = raw_suite_dir
                self._state = PipelineState.PROVISION_SOURCE
                return
            raise PipelineError(
                f"Raw source directory not found: {raw_suite_dir}. "
                "Provision the dataset first, or set "
                "data.source.auto_download=true to download it from the "
                "configured HuggingFace mirror."
            )

        hdf5_files = sorted(raw_suite_dir.glob("*.hdf5"))
        if not hdf5_files:
            if self._auto_download_enabled():
                logger.info(
                    "  Raw source has no .hdf5 files — auto-download enabled, "
                    "provisioning from the configured HuggingFace mirror."
                )
                self._raw_dir = raw_suite_dir
                self._state = PipelineState.PROVISION_SOURCE
                return
            raise PipelineError(
                f"No .hdf5 files found in {raw_suite_dir}. "
                "Provision the dataset first, or set "
                "data.source.auto_download=true to download it from the "
                "configured HuggingFace mirror."
            )
        logger.info(
            "  OK: source '%s' has %d .hdf5 files",
            raw_suite_dir,
            len(hdf5_files),
        )

        self._raw_dir = raw_suite_dir
        self._state = PipelineState.INGEST_AND_STRIP

    def _provision_source(self) -> None:
        """PROVISION_SOURCE: download the raw artifact from the configured mirror.

        Reads ``data.source.huggingface`` (``repo_id``, ``path``, optional
        pinned ``sha256``) and downloads into the raw directory with SHA-256
        verification (pinned value, else the mirror's LFS metadata, else an
        HDF5 sanity check). Proceeds to ingestion only after a verified
        artifact is on disk; an already-verified file is a no-op.
        """
        logger.info("PROVISION_SOURCE: downloading raw dataset artifact…")
        source = self.data_cfg.get("source")
        if source is None:
            raise PipelineError("data.source is not configured.")
        hf_cfg = source.get("huggingface")
        if hf_cfg is None or not hf_cfg.get("repo_id") or not hf_cfg.get("path"):
            raise PipelineError(
                "data.source.auto_download=true requires data.source.huggingface "
                "with repo_id and path (the HuggingFace mirror artifact)."
            )

        dest_dir = self._raw_dir or self._resolve_raw_dir()
        download_hf_file(
            repo_id=str(hf_cfg["repo_id"]),
            path=str(hf_cfg["path"]),
            dest_dir=dest_dir,
            pinned_sha256=str(hf_cfg["sha256"]) if hf_cfg.get("sha256") else None,
        )
        self._state = PipelineState.INGEST_AND_STRIP

    def _ingest_source(self) -> None:
        """Delegate dataset conversion to the pluggable ``data.ingester``.

        The ingester is a Hydra target receiving ``raw_dir`` and returning
        ``(trajectories, task_index)`` where each trajectory carries
        ``state`` (T, S), ``action`` (T, A), ``phase`` (T,) and ``task_id``.
        """
        from hydra.utils import instantiate

        ingester_cfg = self.data_cfg.get("ingester")
        if ingester_cfg is None:
            raise PipelineError(
                "data.ingester is not configured — the source adapter for the "
                "robomimic/robosuite protocol (docs/plan/final_evaluation_plan.md, "
                "Gate 0/1) is not implemented yet. The former benchmark-specific "
                "ingestion stack was archived to legacy/ and must not be used."
            )

        logger.info("INGEST_AND_STRIP: converting raw HDF5 to trajectories…")
        ingester = instantiate(ingester_cfg, raw_dir=self._raw_dir)
        trajectories, task_index = ingester.ingest()

        if not trajectories:
            raise PipelineError("The ingester produced no trajectories.")

        num_phases = None
        labeler_cfg = self.data_cfg.get("phase_labeler")
        if labeler_cfg is not None and labeler_cfg.get("num_phases") is not None:
            num_phases = int(labeler_cfg.num_phases)
        phase_counts = None
        if num_phases is not None:
            phase_counts = np.zeros(num_phases, dtype=np.int64)
        for traj in trajectories:
            if "phase" not in traj:
                raise PipelineError(
                    f"(demo {traj.get('demo_key', '?')}): the ingester did not "
                    "produce phase labels. PhaseForge training requires labels "
                    "generated from permitted low-dimensional observations."
                )
            if num_phases is not None:
                self._validate_phase_labels(traj, traj["phase"], num_phases)
                phase_counts += np.bincount(
                    np.asarray(traj["phase"], dtype=np.int64),
                    minlength=num_phases,
                )

        if phase_counts is not None and np.any(phase_counts == 0):
            missing = np.flatnonzero(phase_counts == 0).tolist()
            if self._model_uses_phase_labels():
                raise PipelineError(
                    f"Phase labels contain no samples for phase(s) {missing} "
                    f"(per-phase sample counts across all trajectories: "
                    f"{phase_counts.tolist()}). "
                    "The selected model consumes phase labels (phase_head or "
                    "top-level num_phases); degenerate labels would silently "
                    "corrupt phase cross-entropy and router centroid "
                    "initialization. Revise the state-only label thresholds "
                    "or the demonstrations before training."
                )
            logger.warning(
                "Phase labels contain no samples for phase(s) %s. "
                "The selected model does not consume phase labels, so the "
                "BC pilot may continue; PhaseForge centroid initialization "
                "must not proceed until Gate 3 is passed.",
                missing,
            )

        self._check_state_dim_consistency(trajectories)
        self._trajectories = trajectories
        self._task_index = task_index or {}
        self._state = PipelineState.NORMALIZE_AND_SAVE

    def _validate_phase_labels(
        self, traj: dict[str, Any], phases: np.ndarray, num_phases: int
    ) -> None:
        """Guard: phase labels must align with the state length and range.

        A length mismatch or out-of-range phase would corrupt training
        (cross-entropy / scatter / bincount) silently. This makes it a loud
        per-trajectory failure instead.
        """
        T = int(traj["state"].shape[0])
        if phases.ndim != 1:
            raise PipelineError(
                f"(demo {traj.get('demo_key', '?')}): phase labels must have "
                f"shape (T,), got {phases.shape}."
            )
        if phases.shape[0] != T:
            raise PipelineError(
                f"(demo {traj.get('demo_key', '?')}): phase labels have length "
                f"{phases.shape[0]} but state has T={T}. Labeler and ingester "
                "disagree on the trajectory length."
            )
        if phases.size and (int(phases.min()) < 0 or int(phases.max()) >= num_phases):
            raise PipelineError(
                f"(demo {traj.get('demo_key', '?')}): phase labels out of range "
                f"[0, {num_phases}): observed [{phases.min()}, {phases.max()}]."
            )

    def _provenance(self, splits: dict[str, list[int]]) -> dict[str, Any]:
        """Build the manifest audit trail for the final cache (Gate 8).

        Includes full-content SHA-256 of every raw HDF5 file (streamed, so
        multi-GB files are fine — one-time cost per data version), the code
        git commit, the state schema, the normalization method, the
        phase-labeler config and the split task names. Enough to prove
        exactly which dataset produced a result.
        """
        split_task_names: dict[str, list[str]] = {}
        split_demo_keys: dict[str, list[str]] = {}
        id_to_name = {v: k for k, v in self._task_index.items()}
        for split_name, indices in splits.items():
            split_task_names[split_name] = sorted(
                {
                    str(
                        self._trajectories[i].get(
                            "task_name",
                            id_to_name.get(
                                int(self._trajectories[i]["task_id"]),
                                self._trajectories[i]["task_id"],
                            ),
                        )
                    )
                    for i in indices
                }
            )
            split_demo_keys[split_name] = sorted(
                str(self._trajectories[i].get("demo_key", f"trajectory_{i}")) for i in indices
            )

        raw_files: list[dict[str, Any]] = []
        if self._raw_dir is not None and self._raw_dir.exists():
            for path in sorted(self._raw_dir.glob("*.hdf5")):
                logger.info("  Provenance: SHA-256 of %s …", path.name)
                stat = path.stat()
                raw_files.append(
                    {
                        "name": path.name,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": sha256_file(path),
                    }
                )

        # Dataset revision: record the source's MANIFEST.json (repo id +
        # pinned commit SHA) verbatim when present.
        dataset_manifest: dict[str, Any] | None = None
        source = self.data_cfg.get("source")
        if source is not None and source.get("dir"):
            manifest_path = Path(
                str(source.get("manifest_path") or Path(str(source["dir"])) / "MANIFEST.json")
            )
            try:
                if manifest_path.exists():
                    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Could not read the source manifest for provenance.",
                    exc_info=True,
                )

        state_schema = {
            "schema_version": str(self.data_cfg.get("schema_version", "")),
            "keys": [
                {"key": str(e["key"]), "dim": int(e["dim"])} for e in self.data_cfg.state_keys
            ],
            "state_dim": int(self.data_cfg.get("state_dim", 0)),
            "action_dim": int(self.data_cfg.get("action_dim", 7)),
            "action_contract": OmegaConf.to_container(
                self.data_cfg.get("action_contract", {}), resolve=True
            ),
        }

        labeler_cfg = self.data_cfg.get("phase_labeler")
        provenance: dict[str, Any] = {
            "dataset_manifest": dataset_manifest,
            "code_git_commit": git_commit(),
            "raw_files": raw_files,
            "state_schema": state_schema,
            "normalization": {
                "strategy": str(self.data_cfg.get("normalization_strategy", "zscore")),
                "ddof": 1,
                "train_split_only": True,
            },
            "sampling": {
                "sequence_length": int(self.data_cfg.get("sequence_length", 1)),
                "stride": int(self.data_cfg.get("stride", 1)),
            },
            "split_task_names": split_task_names,
            "split_demo_keys": split_demo_keys,
            "configuration": OmegaConf.to_yaml(self.data_cfg, resolve=True),
        }
        if labeler_cfg is not None:
            provenance["phase_labeler"] = OmegaConf.to_container(labeler_cfg, resolve=True)
        environment_metadata: list[dict[str, Any]] = []
        seen_metadata: set[str] = set()
        for traj in self._trajectories:
            metadata = traj.get("env_metadata")
            if not isinstance(metadata, dict):
                continue
            serialized = json.dumps(metadata, sort_keys=True)
            if serialized not in seen_metadata:
                seen_metadata.add(serialized)
                environment_metadata.append(metadata)
        provenance["environment_metadata"] = environment_metadata
        return provenance

    def _normalize_and_save(self) -> None:
        logger.info("NORMALIZE_AND_SAVE: computing statistics and persisting cache…")

        # Split before normalization so validation never contributes to the
        # fitted statistics.
        splits = self._build_task_level_splits()

        # Compute normalization stats on TRAIN split only.
        normalizer = RunningStatNormalizer()
        for idx in splits["train"]:
            traj = self._trajectories[idx]
            state_np = traj["state"]  # (T, S) numpy array
            normalizer.update(state_np)

        frozen_norm = normalizer.finalize()

        # Normalize all splits and convert to tensors
        for traj in self._trajectories:
            state_t = torch.from_numpy(traj["state"]).float()
            action_t = torch.from_numpy(traj["action"]).float()
            phase_t = torch.from_numpy(traj["phase"]).long()
            traj["state"] = frozen_norm.normalize(state_t)
            traj["action"] = action_t
            traj["phase"] = phase_t

        norm_stats = {"mean": frozen_norm.mean, "std": frozen_norm.std}

        self.cache_manager.save(
            config_hash=self.config_hash,
            trajectories=self._trajectories,
            norm_stats=norm_stats,
            splits=splits,
            task_index=self._task_index,
            provenance=self._provenance(splits),
        )
        self._norm_stats = norm_stats
        self._splits = splits
        self._state = PipelineState.READY

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _build_task_level_splits(self) -> dict[str, list[int]]:
        """Build the configured task-level or within-task trajectory split.

        ``strategy=task`` is appropriate for a multitask generalization test:
        every task is entirely in train or validation. The primary protocol is
        single-task, so ``strategy=trajectory`` holds out demonstrations from
        each task and prevents the validation loader from silently becoming
        empty.

        Reads ``data.split`` ({strategy, val_ratio, seed}) with safe defaults.

        Seed policy (intentional, documented for the paper): the split RNG
        is seeded from ``data.split.seed`` and **not** from
        ``cfg.project.seed``. A model-seed change must not re-shuffle the
        train/val boundary — otherwise every cell of an ablation seed sweep
        would see a different validation set, breaking direct comparison
        of val curves across cells and introducing a hidden coupling
        between the seed we report and the data we hold out. Across the
        protocol seeds [42, 43, 44] every cell therefore sees the *same*
        train/val trajectories; only the model initialisation and the
        per-epoch DataLoader shuffle (driven by
        :meth:`_train_sampler_generator`) differ.
        """
        split_cfg = self.data_cfg.get("split", {})
        strategy = str(split_cfg.get("strategy", "task"))
        val_ratio = float(split_cfg.get("val_ratio", DEFAULT_VAL_RATIO))
        split_seed = int(split_cfg.get("seed", DEFAULT_SPLIT_SEED))
        if strategy not in {"task", "trajectory"}:
            raise PipelineError(
                f"Unsupported data.split.strategy={strategy!r}; expected 'task' or 'trajectory'."
            )
        if not 0.0 <= val_ratio < 1.0:
            raise PipelineError("data.split.val_ratio must be in [0, 1).")

        # Group trajectory indices by task_id
        by_task: dict[int, list[int]] = defaultdict(list)
        for i, traj in enumerate(self._trajectories):
            by_task[int(traj["task_id"])].append(i)

        task_ids = sorted(by_task.keys())
        rng = np.random.default_rng(split_seed)

        if strategy == "trajectory":
            if bool(split_cfg.get("use_dataset_filters", True)):
                filtered = [
                    (i, traj.get("dataset_split")) for i, traj in enumerate(self._trajectories)
                ]
                if any(split is not None for _, split in filtered):
                    missing = [i for i, split in filtered if split not in {"train", "val"}]
                    if missing:
                        raise PipelineError(
                            "The HDF5 dataset contains an incomplete train/valid "
                            f"filter assignment; missing trajectories: {missing[:8]}"
                        )
                    train_idx = [i for i, split in filtered if split == "train"]
                    val_idx = [i for i, split in filtered if split == "val"]
                    if not train_idx or not val_idx:
                        raise PipelineError("The HDF5 train/valid filters must both be non-empty.")
                    logger.info(
                        "  Dataset-filter split: %d train trajectories, %d val "
                        "trajectories across %d tasks",
                        len(train_idx),
                        len(val_idx),
                        len(task_ids),
                    )
                    return {"train": train_idx, "val": val_idx}

            train_idx: list[int] = []
            val_idx: list[int] = []
            for task_id in task_ids:
                indices = np.asarray(by_task[task_id], dtype=np.int64)
                rng.shuffle(indices)
                if len(indices) < 2:
                    n_val = 0
                else:
                    n_val = (
                        min(
                            len(indices) - 1,
                            max(1, int(round(len(indices) * val_ratio))),
                        )
                        if val_ratio > 0
                        else 0
                    )
                val_idx.extend(int(i) for i in indices[:n_val])
                train_idx.extend(int(i) for i in indices[n_val:])
            logger.info(
                "  Trajectory-level split: %d train trajectories, %d val "
                "trajectories across %d tasks",
                len(train_idx),
                len(val_idx),
                len(task_ids),
            )
            if val_ratio > 0 and task_ids and not val_idx:
                raise PipelineError(
                    "The trajectory split produced no validation trajectories. "
                    "Provide at least two demonstrations for the single-task "
                    "protocol or set val_ratio=0 only for an explicitly "
                    "training-only pilot."
                )
            return {"train": train_idx, "val": val_idx}

        rng.shuffle(task_ids)

        # Bug 5 fix: hard floor so val never collapses to zero tasks.
        n_val_tasks = max(1, round(len(task_ids) * val_ratio)) if task_ids else 0
        # A task-level single-task split has no validation task. Callers that
        # need validation for a single task must select strategy=trajectory.
        if len(task_ids) <= 1:
            n_val_tasks = 0

        val_tasks = set(task_ids[:n_val_tasks])
        train_tasks = set(task_ids[n_val_tasks:])

        train_idx: list[int] = []
        val_idx: list[int] = []
        for tid in task_ids:
            if tid in val_tasks:
                val_idx.extend(by_task[tid])
            else:
                train_idx.extend(by_task[tid])

        logger.info(
            "  Task-level split: %d train tasks (%d trajs), %d val tasks (%d trajs)",
            len(train_tasks),
            len(train_idx),
            len(val_tasks),
            len(val_idx),
        )
        return {"train": train_idx, "val": val_idx}

    def phase_counts(self, split: str = "train") -> dict[int, int]:
        """Per-class phase step counts over one split (for class weighting).

        Counts each labelled step (trajectory timestep) in the requested
        split, keyed by phase index. Only meaningful after ``run()`` has
        reached READY (``self._trajectories``/``self._splits`` populated).
        """
        if split not in self._splits:
            raise PipelineError(f"Unknown split {split!r}; expected one of {sorted(self._splits)}")
        counts: dict[int, int] = defaultdict(int)
        for idx in self._splits[split]:
            phase = np.asarray(self._trajectories[idx]["phase"])
            for p, n in zip(*np.unique(phase, return_counts=True)):
                counts[int(p)] += int(n)
        return counts

    # ------------------------------------------------------------------
    # DataLoader construction
    # ------------------------------------------------------------------

    def _train_sampler_generator(self) -> torch.Generator:
        """Return a CPU ``torch.Generator`` for the train DataLoader sampler.

        The train DataLoader's shuffle uses an explicit generator seeded
        from ``cfg.project.seed`` so the per-epoch sample order is fully
        reproducible from the project seed alone. ``torch.Generator`` is
        placed on CPU because DataLoader worker processes cannot share a
        CUDA generator across the multiprocessing boundary.

        The generator is cached on the instance so a second call (e.g.
        after a cache miss + rerun) returns the same object and consumes
        no additional entropy — important for resume correctness.

        Returns:
            A ``torch.Generator`` on ``cpu`` seeded with the project seed.
        """
        cached = getattr(self, "_train_generator", None)
        if cached is not None:
            return cached
        project_seed = self.cfg.get("project", {}).get("seed", 0)
        try:
            seed_value = int(project_seed)
        except (TypeError, ValueError) as exc:
            raise PipelineError(
                f"cfg.project.seed must be an integer for DataLoader reproducibility, "
                f"got {project_seed!r}."
            ) from exc
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed_value)
        self._train_generator = generator
        return generator

    def _build_dataloaders(self) -> dict[str, DataLoader | None]:
        data_cfg = self.data_cfg
        result: dict[str, DataLoader | None] = {}

        for split_name, indices in self._splits.items():
            if not indices:
                result[split_name] = None
                continue

            split_trajs = [self._trajectories[i] for i in indices]
            is_train = split_name == "train"
            corruption_rate = (
                float(data_cfg.get("phase_corruption_rate", 0.0)) if is_train else 0.0
            )
            default_seed = int(self.cfg.get("project", {}).get("seed", 42))
            corruption_seed = (
                int(data_cfg.get("phase_corruption_seed", default_seed))
                if is_train
                else 42
            )
            labeler_cfg = data_cfg.get("phase_labeler")
            num_phases_val = int(labeler_cfg.get("num_phases", 6)) if labeler_cfg else 6
            phase_shuffle = (
                bool(data_cfg.get("phase_shuffle_control", False)) if is_train else False
            )

            dataset = StateOnlyDataset(
                trajectories=split_trajs,
                sequence_length=int(data_cfg.sequence_length),
                stride=int(data_cfg.stride),
                phase_corruption_rate=corruption_rate,
                phase_corruption_seed=corruption_seed,
                num_phases=num_phases_val,
                phase_shuffle_control=phase_shuffle,
            )

            # Cap num_workers to os.cpu_count() to prevent warnings and slowdowns.
            # The dataset is fully materialized in memory, so zero remains a
            # valid choice for small CPU-only pilots where worker IPC costs
            # more than it saves.
            num_workers = max(0, int(data_cfg.num_workers))
            if hasattr(os, "cpu_count") and os.cpu_count() is not None:
                num_workers = min(num_workers, os.cpu_count())

            # These options are intentionally derived from the effective
            # runtime. Pinned memory is useful for CUDA host-to-device copies,
            # but adds overhead on CPU-only runs. prefetch_factor and
            # persistent_workers are only legal when workers are enabled.
            project_cfg = self.cfg.get("project")
            requested_device = str(
                project_cfg.get("device", "cuda") if project_cfg is not None else "cuda"
            )
            pin_memory = (
                bool(data_cfg.get("pin_memory", False))
                and requested_device.startswith("cuda")
                and torch.cuda.is_available()
            )
            loader_options: dict[str, Any] = {
                "pin_memory": pin_memory,
            }
            if num_workers > 0:
                loader_options["prefetch_factor"] = max(1, int(data_cfg.get("prefetch_factor", 2)))
                loader_options["persistent_workers"] = bool(
                    data_cfg.get("persistent_workers", False)
                )

            loader = DataLoader(
                dataset,
                batch_size=int(data_cfg.batch_size),
                shuffle=is_train,
                num_workers=num_workers,
                collate_fn=PhaseAwareCollator(),
                drop_last=is_train,
                generator=self._train_sampler_generator() if is_train else None,
                **loader_options,
            )
            result[split_name] = loader
            logger.info(f"  {split_name}: {len(dataset)} samples, {len(loader)} batches")

        return result

    def _check_state_dim_consistency(self, trajs: list[dict[str, Any]]) -> None:
        """Guard: every decoded state vector must match ``data.state_dim``.

        ``data.state_dim`` sizes the model encoder's ``input_dim``. A
        mismatch — e.g. the ingester producing a different width than the
        config — would otherwise surface only as a shape error at the first
        model forward pass, after a useless re-ingest. Failing loud here
        catches it at the source.
        """
        expected = int(self.data_cfg.get("state_dim", 0))
        if expected <= 0:
            return
        bad = sorted(
            {int(traj["state"].shape[-1]) for traj in trajs if traj["state"].shape[-1] != expected}
        )
        if bad:
            raise PipelineError(
                f"Decoded state dims {bad} do not match data.state_dim="
                f"{expected}. Fix data.source / data.ingester / data.state_dim, "
                "then re-ingest — the model encoder input_dim would disagree "
                "with the decoded states."
            )
