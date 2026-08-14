"""Environment fingerprint for one run (adapted from ``csd_observer``).

Captures the dependency versions, git state, host info, and dataset-cache
identity that together make a single run reproducible from its output
directory alone. ``run_writer.write_environment()`` persists the dict
returned here as ``<run_dir>/metadata/environment.json``.
"""

from __future__ import annotations

import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any

from phaseforge.utils.config import git_info


def _safe_version(modname: str) -> str | None:
    """Import ``modname`` defensively and return its ``__version__``.

    Returns ``None`` if the module is not installed or has no
    ``__version__`` attribute; never raises.
    """
    try:
        module = __import__(modname)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def collect_environment(
    *,
    data_config_hash: str | None = None,
    config_hash: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single ``environment.json`` payload.

    Args:
        data_config_hash: Pipeline-level data config hash (the value
            printed by the state machine). The data hash is the cache
            identity; the full ``config_hash`` only matches when the
            data subset is included.
        config_hash: Full ``cfg`` config hash, written by
            :func:`phaseforge.utils.config.config_hash`.
        extra: Caller-supplied extras (e.g. resolved device, git branch).
    """
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "user": os.environ.get("USER") or os.environ.get("USERNAME") or "",
        "packages": {
            "torch": _safe_version("torch"),
            "numpy": _safe_version("numpy"),
            "hydra-core": _safe_version("hydra"),
            "omegaconf": _safe_version("omegaconf"),
            "scikit-learn": _safe_version("sklearn"),
            "scipy": _safe_version("scipy"),
            "h5py": _safe_version("h5py"),
            "filelock": _safe_version("filelock"),
            "wandb": _safe_version("wandb"),
        },
        "git_sha": git_info()["commit"],
        "git_branch": git_info()["branch"],
        "data_config_hash": data_config_hash,
        "config_hash": config_hash,
    }
    if extra:
        env["extra"] = extra
    return env


__all__ = ["collect_environment"]
