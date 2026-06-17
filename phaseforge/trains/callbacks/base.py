"""Callback base class."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from phaseforge.trains.loops.base import BaseTrainer


class Callback:
    """Base class for all training callbacks."""

    def on_train_start(self, trainer: BaseTrainer) -> None:
        pass

    def on_epoch_start(self, trainer: BaseTrainer) -> None:
        pass

    def on_train_step(self, trainer: BaseTrainer, step: int, metrics: dict[str, float]) -> None:
        pass

    def on_epoch_end(self, trainer: BaseTrainer, val_metrics: dict[str, float]) -> None:
        pass

    def on_train_end(self, trainer: BaseTrainer) -> None:
        pass
