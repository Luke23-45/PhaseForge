"""Data ingestion state machine.

This is the single entry point for all data loading. The trainer calls
``DataPipelineStateMachine(cfg).run()`` and receives a dict of DataLoaders.

States
------
CHECK_PERSISTENT_CACHE → (hit) READY
                       → (miss) DOWNLOAD_SOURCE → INGEST_AND_STRIP
                                                → NORMALIZE_AND_SAVE → READY
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from phaseforge.data.ingestion.cache_manager import CacheManager
from phaseforge.data.ingestion.states import PipelineState
from phaseforge.data.common.dataset import StateOnlyDataset
from phaseforge.data.common.collator import PhaseAwareCollator
from phaseforge.data.common.normalizer import RunningStatNormalizer, FrozenNormalizer

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when the pipeline reaches the ERROR state."""


class DataPipelineStateMachine:
    """Autonomous data pipeline implemented as a finite state machine.

    Args:
        cfg: Root Hydra config. The pipeline uses ``cfg.data`` and
             ``cfg.project`` sub-trees.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.project_cfg = cfg.project

        output_dir = Path(self.project_cfg.output_dir)
        cache_root = output_dir / self.data_cfg.cache_root

        self.cache_manager = CacheManager(cache_root)
        self.config_hash = CacheManager.compute_hash(self.data_cfg)

        self._state = PipelineState.CHECK_PERSISTENT_CACHE
        self._error_msg: str = ""

        # Populated during processing
        self._trajectories: list[dict[str, Any]] = []
        self._norm_stats: dict[str, torch.Tensor] = {}
        self._splits: dict[str, list[int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, DataLoader]:
        """Execute the FSM to the READY terminal state.

        Returns:
            Dict of ``{"train": DataLoader, "val": DataLoader, "test": DataLoader}``.
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
            elif self._state == PipelineState.DOWNLOAD_SOURCE:
                self._download_source()
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
            self._trajectories, self._norm_stats, self._splits = (
                self.cache_manager.load(self.config_hash)
            )
            self._state = PipelineState.READY
        else:
            logger.info("Cache miss. Proceeding to download.")
            self._state = PipelineState.DOWNLOAD_SOURCE

    def _download_source(self) -> None:
        logger.info("DOWNLOAD_SOURCE: checking / fetching raw data…")
        libero_cfg = self.data_cfg.libero
        raw_dir = Path(self.project_cfg.output_dir) / libero_cfg.local.raw_dir
        raw_dir.mkdir(parents=True, exist_ok=True)

        from phaseforge.data.ingestion._downloader import download_files
        download_files(
            files=list(libero_cfg.remote.files),
            base_url=str(libero_cfg.remote.base_url),
            dest_dir=raw_dir,
        )
        self._raw_dir = raw_dir
        self._state = PipelineState.INGEST_AND_STRIP

    def _ingest_and_strip(self) -> None:
        logger.info("INGEST_AND_STRIP: parsing HDF5, stripping vision, labeling phases…")
        import glob

        from phaseforge.data.libero.vision_stripper import VisionStripper
        from phaseforge.data.libero.phase_labeler import RuleBasedPhaseLabeler
        from hydra.utils import instantiate

        stripper = VisionStripper(state_keys=list(self.data_cfg.state_keys))
        labeler: RuleBasedPhaseLabeler = instantiate(self.data_cfg.libero.phase_labeler)

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
        self._trajectories = all_trajs
        self._state = PipelineState.NORMALIZE_AND_SAVE

    def _normalize_and_save(self) -> None:
        logger.info("NORMALIZE_AND_SAVE: computing statistics and persisting cache…")
        split_cfg = self.data_cfg.libero.split

        # Build index splits
        n = len(self._trajectories)
        indices = list(range(n))

        rng = np.random.default_rng(int(split_cfg.seed))
        rng.shuffle(indices)

        train_end = int(n * split_cfg.train_ratio)
        val_end = train_end + int(n * split_cfg.val_ratio)

        splits = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }

        # Compute normalization stats on TRAIN split only
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
        )
        self._norm_stats = norm_stats
        self._splits = splits
        self._state = PipelineState.READY

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
