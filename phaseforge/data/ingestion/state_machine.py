"""Data ingestion state machine.

This is the single entry point for all data loading. The trainer calls
``DataPipelineStateMachine(cfg).run()`` and receives a dict of DataLoaders.

States
------
CHECK_PERSISTENT_CACHE → (hit) READY
                       → (miss) VALIDATE_SOURCE → INGEST_AND_STRIP
                                                → NORMALIZE_AND_SAVE → READY

Design notes (bugs fixed here, each proven by simulation)
---------------------------------------------------------
- Bug 1: No more fictional box.com download. The FSM consumes pre-downloaded
  data from the env-var-aware data root (paths.py). VALIDATE_SOURCE checks the
  official LIBERO file count (90 / 10) before ingesting.
- Bug 4: The processed cache lives under {data_root}/processed/cache, NOT under
  the per-run outputs/ directory, so the config-hash cache is reused across runs.
- Bug 5: Splits are done at the TASK level (no same-task leakage between train
  and val), and val is guaranteed non-empty via a floor of max(1, ...). LIBERO-LONG
  (role=eval) is exposed separately via get_eval_loader().
- Bug 2: task_id is deterministic (sorted-name -> int) via task_index.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from phaseforge.data.common.collator import PhaseAwareCollator
from phaseforge.data.common.dataset import StateOnlyDataset
from phaseforge.data.common.normalizer import RunningStatNormalizer
from phaseforge.data.ingestion.cache_manager import CacheManager
from phaseforge.data.ingestion.states import PipelineState
from phaseforge.data.libero.object_state import ObjectIndex
from phaseforge.data.libero.task_index import build_task_index
from phaseforge.data.paths import (
    EXPECTED_FILE_COUNTS,
    libero_object_index_path,
    libero_suite_dir,
    processed_cache_root,
)

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when the pipeline reaches the ERROR state."""


class DataPipelineStateMachine:
    """Autonomous data pipeline implemented as a finite state machine.

    Args:
        cfg: Root Hydra config. The pipeline uses ``cfg.data`` and
             ``cfg.data.libero``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.libero_cfg = cfg.data.libero

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

        # Eval (LIBERO-LONG) loader is built lazily and kept out of run()'s
        # return dict because cli.py only reads "train"/"val".
        self._eval_loader: DataLoader | None = None
        self._eval_loaded: bool = False

    # ------------------------------------------------------------------
    # Config consistency
    # ------------------------------------------------------------------

    def _check_num_phases_consistency(self) -> None:
        """Guard against the latent num_phases mismatch bug.

        The integer num_phases is repeated in:
        - data.libero.phase_labeler.num_phases  (label generation)
        - models.phase_head.num_phases          (classifier width)
        - models.router.num_experts             (router width)
        - models.num_phases                     (oracle model)

        Nothing validated they match. If they diverge, the labeler produces
        values the classifier/scatter/bincount cannot index. This guard makes
        the inconsistency a loud, early failure instead of a silent corruption.

        Only checks model fields that exist in the current config (so a
        config that omits the oracle's top-level num_phases doesn't trip it).
        """
        data_phases = int(self.libero_cfg.phase_labeler.num_phases)

        models_cfg = self.cfg.get("models")
        if models_cfg is None:
            return  # model config not part of this run (e.g. eval-only)

        candidates: list[tuple[str, int]] = []
        phase_head = models_cfg.get("phase_head")
        if phase_head is not None and phase_head.get("num_phases") is not None:
            candidates.append(("models.phase_head.num_phases", int(phase_head.num_phases)))

        router = models_cfg.get("router")
        if router is not None and router.get("num_experts") is not None:
            candidates.append(("models.router.num_experts", int(router.num_experts)))

        # Oracle model carries a top-level num_phases; other models may not.
        if models_cfg.get("num_phases") is not None:
            candidates.append(("models.num_phases", int(models_cfg.num_phases)))

        mismatches = [
            (name, val) for name, val in candidates if val != data_phases
        ]
        if mismatches:
            details = ", ".join(f"{n}={v}" for n, v in mismatches)
            raise PipelineError(
                f"num_phases inconsistency: data.libero.phase_labeler.num_phases="
                f"{data_phases} but {details}. All must match, otherwise the "
                "phase labels cannot be indexed by the classifier/router/scatter."
            )

    # ------------------------------------------------------------------
    # Object-state channel (P-Stage 1)
    # ------------------------------------------------------------------

    def _load_object_index(self) -> ObjectIndex | None:
        """Load the per-task object decode tables, or None when disabled.

        Reads ``data.object_state``; the index file defaults to
        ``{data_root}/raw/libero/object_index.json`` (census output) unless
        ``data.object_state.index_path`` is set explicitly.

        Validates the config <-> index handshake: the config's
        ``k_slots`` / ``dim_per_object`` / ``include_mask`` and the resulting
        ``data.state_dim`` must match what the index actually produces.
        Otherwise the decoded state vectors, the normalizer's mask exclusion
        and the model encoder's ``input_dim`` would silently disagree.
        """
        oscfg = self.data_cfg.get("object_state")
        if oscfg is None or not oscfg.get("enabled", True):
            return None
        raw_path = oscfg.get("index_path")
        path = Path(raw_path) if raw_path else libero_object_index_path()
        index = ObjectIndex.load(path)

        cfg_k_slots = oscfg.get("k_slots")
        if cfg_k_slots is not None and int(cfg_k_slots) != index.k_slots:
            raise ValueError(
                f"data.object_state.k_slots={cfg_k_slots} does not match the "
                f"index at {path}: k_slots={index.k_slots}. Rebuild the index "
                "with build_object_index or update the config, then re-ingest."
            )
        cfg_dim = oscfg.get("dim_per_object")
        if cfg_dim is not None and int(cfg_dim) != index.dim_per_object:
            raise ValueError(
                f"data.object_state.dim_per_object={cfg_dim} does not match "
                f"the index at {path}: dim_per_object={index.dim_per_object}."
            )
        cfg_include_mask = oscfg.get("include_mask", True)
        if bool(cfg_include_mask) != index.include_mask:
            raise ValueError(
                f"data.object_state.include_mask={cfg_include_mask} does not "
                f"match the index at {path}: include_mask={index.include_mask}."
            )

        # The model encoder is sized from data.state_dim; the decoded states
        # are sized from the index. They must agree.
        state_dim = self.data_cfg.get("state_dim")
        if state_dim is not None:
            proprio_dim = sum(int(e["dim"]) for e in self.data_cfg.state_keys)
            expected = proprio_dim + index.state_block_size
            if int(state_dim) != expected:
                raise ValueError(
                    f"data.state_dim={state_dim} does not match the object "
                    f"layout: proprio {proprio_dim} + object-state block "
                    f"{index.state_block_size} = {expected}. The model "
                    "encoder input_dim would disagree with the decoded states."
                )
        return index

    def _check_state_dim_consistency(self, trajs: list[dict[str, Any]]) -> None:
        """Guard: every decoded state vector must match ``data.state_dim``.

        ``data.state_dim`` sizes the model encoder's ``input_dim``, while the
        decoded width is determined by the object index (k_slots, mask flag)
        and by whether the object channel is enabled. A mismatch — e.g. the
        channel disabled but a stale ``state_dim=151``, or index/config drift
        — would otherwise surface only as a shape error at the first model
        forward pass, after a useless 66GB re-ingest. Failing loud here
        catches it at the source.
        """
        expected = int(self.data_cfg.get("state_dim", 0))
        if expected <= 0:
            return
        bad = sorted(
            {
                int(traj["state"].shape[-1])
                for traj in trajs
                if traj["state"].shape[-1] != expected
            }
        )
        if bad:
            raise PipelineError(
                f"Decoded state dims {bad} do not match data.state_dim="
                f"{expected}. Fix data.object_state / the object index / "
                "data.state_dim, then re-ingest — the model encoder "
                "input_dim would disagree with the decoded states."
            )

    def _proprio_slices(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Derive the labeler's EEF/gripper slices from the configured state_keys.

        Single source of truth: the proprioceptive block is the concatenation
        of ``state_keys`` (in config order), and the object block is appended
        after it, so these slices are stable regardless of the object channel.
        """
        cursor = 0
        by_key: dict[str, tuple[int, int]] = {}
        for entry in self.data_cfg.state_keys:
            key = entry["key"]
            dim = int(entry["dim"])
            by_key[key] = (cursor, cursor + dim)
            cursor += dim
        return by_key["robot0_eef_pos"], by_key["robot0_gripper_qpos"]

    def _object_mask_dims(self, object_index: ObjectIndex | None) -> set[int] | None:
        """State indices of the binary mask dims (excluded from normalization)."""
        if object_index is None or not object_index.include_mask:
            return None
        proprio_dim = sum(int(e["dim"]) for e in self.data_cfg.state_keys)
        obj_start = proprio_dim + object_index.k_slots * object_index.dim_per_object
        return set(range(obj_start, obj_start + object_index.k_slots))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, DataLoader]:
        """Execute the FSM to the READY terminal state.

        Returns:
            Dict of ``{"train": DataLoader, "val": DataLoader}``.
            For a ``role=eval`` suite config this returns ``{}`` and the
            data is instead available via :meth:`get_eval_loader`.
            Splits that have 0 samples return None in their slot.
        """
        logger.info(f"Pipeline starting. Config hash: {self.config_hash}")

        while self._state not in (PipelineState.READY, PipelineState.ERROR):
            self._step()

        if self._state == PipelineState.ERROR:
            raise PipelineError(self._error_msg)

        return self._build_dataloaders()

    def get_eval_loader(self) -> DataLoader | None:
        """Build and return the LIBERO-LONG evaluation DataLoader.

        Returns None if the configured suite is not ``role=eval``. The
        loader is built once and cached. This is intentionally separate
        from :meth:`run` so the train pipeline never accidentally mixes
        in evaluation data.
        """
        if self._eval_loaded:
            return self._eval_loader
        self._eval_loaded = True

        role = str(self.libero_cfg.get("role", "train"))
        if role != "eval":
            self._eval_loader = None
            return None

        # Load + strip LIBERO-LONG without normalizing (eval uses the
        # train-frozen normalizer). We ingest into a throwaway buffer.
        eval_trajectories = self._ingest_suite_into(self.libero_cfg.suite)
        frozen_norm = self._load_or_build_normalizer()
        for traj in eval_trajectories:
            state_t = torch.from_numpy(traj["state"]).float()
            traj["state"] = frozen_norm.normalize(state_t)
            traj["action"] = torch.from_numpy(traj["action"]).float()
            traj["phase"] = torch.from_numpy(traj["phase"]).long()

        if not eval_trajectories:
            self._eval_loader = None
            return None

        dataset = StateOnlyDataset(
            trajectories=eval_trajectories,
            sequence_length=int(self.data_cfg.sequence_length),
            stride=int(self.data_cfg.stride),
        )
        self._eval_loader = DataLoader(
            dataset,
            batch_size=int(self.data_cfg.batch_size),
            shuffle=False,
            num_workers=int(self.data_cfg.num_workers),
            pin_memory=bool(self.data_cfg.pin_memory),
            collate_fn=PhaseAwareCollator(),
            drop_last=False,
        )
        logger.info(f"  eval: {len(dataset)} samples, {len(self._eval_loader)} batches")
        return self._eval_loader

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _step(self) -> None:
        try:
            if self._state == PipelineState.CHECK_PERSISTENT_CACHE:
                self._check_cache()
            elif self._state == PipelineState.VALIDATE_SOURCE:
                self._validate_source()
            elif self._state == PipelineState.INGEST_AND_STRIP:
                self._ingest_and_strip()
            elif self._state == PipelineState.NORMALIZE_AND_SAVE:
                self._normalize_and_save()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline error")
            self._error_msg = str(exc)
            self._state = PipelineState.ERROR

    def _check_cache(self) -> None:
        logger.info("CHECK_PERSISTENT_CACHE: looking for cached data…")
        if self.cache_manager.cache_exists(self.config_hash):
            logger.info("Cache hit. Loading from disk.")
            (
                self._trajectories,
                self._norm_stats,
                self._splits,
                self._task_index,
            ) = self.cache_manager.load(self.config_hash)
            self._state = PipelineState.READY
        else:
            logger.info("Cache miss. Proceeding to validate source.")
            self._state = PipelineState.VALIDATE_SOURCE

    def _validate_source(self) -> None:
        """Bug 1 fix: consume pre-downloaded data and verify file count.

        Replaces the old ``_download_source`` that called fictional
        box.com URLs. We look for the suite folder under the env-var-aware
        data root and apply the official LIBERO file-count integrity check.
        """
        logger.info("VALIDATE_SOURCE: checking pre-downloaded raw data…")
        suite = str(self.libero_cfg.suite)
        raw_suite_dir = libero_suite_dir(suite)

        if not raw_suite_dir.exists():
            raise PipelineError(
                f"Raw suite directory not found: {raw_suite_dir}. "
                "Download the data first with: "
                "python -m phaseforge.data.scripts.download_libero"
            )

        # Official LIBERO integrity check = file count (see paths.EXPECTED_FILE_COUNTS).
        actual = len(list(raw_suite_dir.glob("*.hdf5")))
        expected = EXPECTED_FILE_COUNTS[suite]
        if actual != expected:
            raise PipelineError(
                f"Integrity check FAILED for suite '{suite}': expected "
                f"{expected} .hdf5 files, found {actual} in {raw_suite_dir}. "
                "Re-run: python -m phaseforge.data.scripts.download_libero"
            )
        logger.info(
            "  OK: suite '%s' has %d .hdf5 files at %s",
            suite, actual, raw_suite_dir,
        )

        # Bug 2: build the deterministic task index once, here.
        self._task_index = build_task_index(raw_suite_dir)
        self._raw_dir = raw_suite_dir
        self._state = PipelineState.INGEST_AND_STRIP

    def _ingest_and_strip(self) -> None:
        logger.info("INGEST_AND_STRIP: parsing HDF5, stripping vision, labeling phases…")
        from hydra.utils import instantiate

        from phaseforge.data.libero.phase_labeler import RuleBasedPhaseLabeler
        from phaseforge.data.libero.vision_stripper import VisionStripper

        object_index = self._load_object_index()
        stripper = VisionStripper(
            state_keys=list(self.data_cfg.state_keys),
            task_index=self._task_index,
            object_index=object_index,
        )
        eef_slice, gripper_slice = self._proprio_slices()
        labeler: RuleBasedPhaseLabeler = instantiate(
            self.data_cfg.libero.phase_labeler,
            eef_pos_slice=eef_slice,
            gripper_qpos_slice=gripper_slice,
        )

        hdf5_files = sorted(self._raw_dir.glob("*.hdf5"))
        if not hdf5_files:
            raise PipelineError(f"No .hdf5 files found in {self._raw_dir}")

        all_trajs: list[dict[str, Any]] = []
        for hdf5_path in hdf5_files:
            logger.info(f"  Parsing {hdf5_path.name}")
            stripped = stripper.strip(hdf5_path)
            for traj in stripped:
                phase_labels = labeler.label(traj)
                traj["phase"] = phase_labels
                all_trajs.append(traj)

        logger.info(f"  Total trajectories after stripping: {len(all_trajs)}")
        self._check_state_dim_consistency(all_trajs)
        self._trajectories = all_trajs
        self._state = PipelineState.NORMALIZE_AND_SAVE

    def _normalize_and_save(self) -> None:
        logger.info("NORMALIZE_AND_SAVE: computing statistics and persisting cache…")
        split_cfg = self.data_cfg.libero.split

        # Bug 5 fix: TASK-LEVEL split (no same-task leakage).
        splits = self._build_task_level_splits(split_cfg)

        # Compute normalization stats on TRAIN split only.
        # Mask dims (object-slot occupancy, already binary) are excluded.
        object_index = self._load_object_index()
        normalizer = RunningStatNormalizer(
            ignore_dims=self._object_mask_dims(object_index)
        )
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
        )
        self._norm_stats = norm_stats
        self._splits = splits
        self._state = PipelineState.READY

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _build_task_level_splits(
        self, split_cfg: DictConfig
    ) -> dict[str, list[int]]:
        """Group trajectories by task_id and split at the task level.

        Guarantees:
        - No same-task leakage: every task's demos land entirely in train
          OR entirely in val, never both.
        - val is never empty for a train-role suite: a hard floor of
          ``max(1, round(n_tasks * val_ratio))`` replaces the old
          ``int(n * val_ratio)`` which rounded small val sets to 0.

        For a role=eval suite, all trajectories go to "eval" and train/val
        are empty (the data is exposed via get_eval_loader instead).
        """
        role = str(self.libero_cfg.get("role", "train"))
        if role == "eval":
            return {"train": [], "val": [], "eval": list(range(len(self._trajectories)))}

        # Group trajectory indices by task_id
        by_task: dict[int, list[int]] = defaultdict(list)
        for i, traj in enumerate(self._trajectories):
            by_task[int(traj["task_id"])].append(i)

        task_ids = sorted(by_task.keys())
        rng = np.random.default_rng(int(split_cfg.seed))
        rng.shuffle(task_ids)

        val_ratio = float(split_cfg.val_ratio)
        # Bug 5 fix: hard floor so val never collapses to zero tasks.
        n_val_tasks = max(1, round(len(task_ids) * val_ratio)) if task_ids else 0
        # Edge case: if only one task exists, keep it in train.
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
            len(train_tasks), len(train_idx),
            len(val_tasks), len(val_idx),
        )
        return {"train": train_idx, "val": val_idx, "eval": []}

    # ------------------------------------------------------------------
    # Normalizer sharing (train -> eval)
    # ------------------------------------------------------------------

    def _load_or_build_normalizer(self):
        """Return a frozen normalizer for the eval split.

        Per the proposal, evaluation must use the TRAIN-frozen statistics
        (never compute stats from eval data). We load them from the
        processed cache if present; otherwise build from any cached train
        data. If neither exists we raise — eval cannot invent its own stats.
        """
        if self._norm_stats:
            from phaseforge.data.common.normalizer import FrozenNormalizer
            return FrozenNormalizer(mean=self._norm_stats["mean"], std=self._norm_stats["std"])

        if self.cache_manager.cache_exists(self.config_hash):
            _, norm_stats, _, _ = self.cache_manager.load(self.config_hash)
            from phaseforge.data.common.normalizer import FrozenNormalizer
            return FrozenNormalizer(mean=norm_stats["mean"], std=norm_stats["std"])

        raise PipelineError(
            "Cannot build eval normalizer: no cached train statistics found. "
            "Run the train-role pipeline first so the normalizer is persisted."
        )

    def _ingest_suite_into(self, suite: str) -> list[dict[str, Any]]:
        """Ingest a suite folder into a fresh trajectory list (no caching)."""
        from hydra.utils import instantiate

        from phaseforge.data.libero.phase_labeler import RuleBasedPhaseLabeler
        from phaseforge.data.libero.vision_stripper import VisionStripper

        raw_dir = libero_suite_dir(suite)
        task_index = build_task_index(raw_dir)
        object_index = self._load_object_index()
        stripper = VisionStripper(
            state_keys=list(self.data_cfg.state_keys),
            task_index=task_index,
            object_index=object_index,
        )
        eef_slice, gripper_slice = self._proprio_slices()
        labeler: RuleBasedPhaseLabeler = instantiate(
            self.data_cfg.libero.phase_labeler,
            eef_pos_slice=eef_slice,
            gripper_qpos_slice=gripper_slice,
        )

        trajs: list[dict[str, Any]] = []
        for hdf5_path in sorted(raw_dir.glob("*.hdf5")):
            for traj in stripper.strip(hdf5_path):
                traj["phase"] = labeler.label(traj)
                trajs.append(traj)
        self._check_state_dim_consistency(trajs)
        return trajs

    # ------------------------------------------------------------------
    # DataLoader construction
    # ------------------------------------------------------------------

    def _build_dataloaders(self) -> dict[str, DataLoader | None]:
        data_cfg = self.data_cfg
        result: dict[str, DataLoader | None] = {}

        for split_name, indices in self._splits.items():
            if not indices:
                result[split_name] = None
                continue

            split_trajs = [self._trajectories[i] for i in indices]
            dataset = StateOnlyDataset(
                trajectories=split_trajs,
                sequence_length=int(data_cfg.sequence_length),
                stride=int(data_cfg.stride),
            )
            is_train = split_name == "train"
            loader = DataLoader(
                dataset,
                batch_size=int(data_cfg.batch_size),
                shuffle=is_train,
                num_workers=int(data_cfg.num_workers),
                pin_memory=bool(data_cfg.pin_memory),
                collate_fn=PhaseAwareCollator(),
                drop_last=is_train,
            )
            result[split_name] = loader
            logger.info(
                f"  {split_name}: {len(dataset)} samples, {len(loader)} batches"
            )

        return result
