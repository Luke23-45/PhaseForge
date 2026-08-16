# Data preparation for the complete five-task state-only protocol.
# Run once after provisioning all five HDF5 files.

import json
import os
from pathlib import Path

import h5py
import numpy as np
import phaseforge
from hydra import compose, initialize_config_module

# Use the external data root used by the provisioning cell.
DATA_ROOT = Path("C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/data").resolve()


print(f"PHASEFORGE_DATA_DIR: {DATA_ROOT}")
print(f"phaseforge imported from: {phaseforge.__file__}")

TASKS = ("lift", "can", "square", "tool_hang", "transport")

# Labeler self-test: verifies the installed checkout is current.
from phaseforge.data.robomimic.phase_labeler import RuleBasedPhaseLabeler

_st = np.zeros((40, 9), dtype=np.float32)
_st[:, 7] = np.concatenate([np.full(20, 0.0208), np.full(20, 0.04)])
_st[:, 8] = -_st[:, 7]
_st[:, 0] = np.linspace(0.0, 0.8, 40)

_ph = RuleBasedPhaseLabeler().label({"state": _st})
if len(set(_ph.tolist())) < 3:
    raise RuntimeError(
        "The installed phaseforge checkout is stale: the phase labeler "
        "cannot separate open/closed gripper states. Restart the runtime, "
        "refresh the repository, reinstall phaseforge, and rerun this cell."
    )

print(f"Labeler self-test OK ({len(set(_ph.tolist()))} phases)")

from phaseforge.data.ingestion.state_machine import DataPipelineStateMachine

prepared = {}

with initialize_config_module(
    version_base="1.3",
    config_module="phaseforge.config",
):
    for task in TASKS:
        print(f"\nPreparing task: {task}")

        cfg = compose(
            config_name="main",
            overrides=[
                f"data={task}",
                "models=phaseforge",
                "train=stage1",
                "data.source.auto_download=false",
            ],
        )

        raw_dir = Path(str(cfg.data.source.dir))
        if not raw_dir.is_dir():
            raise RuntimeError(
                f"{task}: source directory is missing: {raw_dir}. "
                "Run the verified five-task provisioning cell first."
            )

        hdf5_files = sorted(raw_dir.glob("*.hdf5"))
        if not hdf5_files:
            raise RuntimeError(f"{task}: no HDF5 file found in {raw_dir}")

        # Validate the raw robomimic container before ingestion.
        for hdf5_path in hdf5_files:
            with h5py.File(hdf5_path, "r") as h5:
                if "data" not in h5:
                    raise RuntimeError(f"{task}: {hdf5_path} has no 'data' group")
                if "env_args" not in h5["data"].attrs:
                    raise RuntimeError(
                        f"{task}: {hdf5_path} is missing data.attrs['env_args']"
                    )

        print(f"Source OK: {raw_dir} ({len(hdf5_files)} HDF5 file(s))")

        pipeline = DataPipelineStateMachine(cfg=cfg)
        dataloaders = pipeline.run()

        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]

        prepared[task] = {
            "cfg": cfg,
            "pipeline": pipeline,
            "dataloaders": dataloaders,
        }

        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples:   {len(val_loader.dataset)}")
        print(
            f"Schema: state_dim={cfg.data.state_dim}, "
            f"action_dim={cfg.data.action_dim}, "
            f"phases={cfg.data.phase_labeler.num_phases}"
        )
        print(f"Cache key: {pipeline.config_hash}")

# Preserve compatibility with existing Lift-only notebook method cells.
cfg = prepared["lift"]["cfg"]
pipeline = prepared["lift"]["pipeline"]
dataloaders = prepared["lift"]["dataloaders"]
train_loader = dataloaders["train"]
val_loader = dataloaders["val"]

print("\nAll five task caches are ready.")
print("Lift remains selected as the default notebook task.")
print("Use prepared[task] when running a method for another task.")