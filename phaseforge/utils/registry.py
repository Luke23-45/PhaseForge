"""Factory registry: build models, data pipelines, and trainers via Hydra instantiate."""

from __future__ import annotations

from omegaconf import DictConfig
from hydra.utils import instantiate


def build_model(cfg: DictConfig):
    """Instantiate the model from the models config subtree.

    The model _target_ must resolve to a subclass of BaseManipulationModel.
    """
    from phaseforge.models.base import BaseManipulationModel

    model = instantiate(cfg.models)
    if not isinstance(model, BaseManipulationModel):
        raise TypeError(
            f"Model {type(model).__name__} does not implement BaseManipulationModel. "
            "Check the _target_ in your model config."
        )
    return model


def build_data_pipeline(cfg: DictConfig):
    """Instantiate and return the data pipeline state machine."""
    from phaseforge.data.ingestion.state_machine import DataPipelineStateMachine

    return DataPipelineStateMachine(cfg=cfg)


def build_trainer(cfg: DictConfig, model, train_loader, val_loader):
    """Instantiate the correct trainer based on cfg.train.stage."""
    stage = cfg.train.get("stage", 1)
    if stage == 1:
        from phaseforge.trains.loops.stage1_loop import Stage1Trainer
        return Stage1Trainer(cfg=cfg, model=model, train_loader=train_loader, val_loader=val_loader)
    elif stage == 2:
        from phaseforge.trains.loops.stage2_loop import Stage2Trainer
        return Stage2Trainer(cfg=cfg, model=model, train_loader=train_loader, val_loader=val_loader)
    else:
        raise ValueError(f"Unknown training stage: {stage}. Must be 1 or 2.")
