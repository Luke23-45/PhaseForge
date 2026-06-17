"""Rollout evaluator (placeholder for LIBERO environment integration)."""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig

from phaseforge.models.base import BaseManipulationModel

logger = logging.getLogger(__name__)


class RolloutEvaluator:
    """Environment-based evaluator."""

    def __init__(self, cfg: DictConfig, model: BaseManipulationModel) -> None:
        self.cfg = cfg
        self.model = model

    def run(self) -> dict[str, float]:
        """Execute environment rollouts."""
        logger.warning("RolloutEvaluator is a placeholder. LIBERO env integration required.")
        return {}
