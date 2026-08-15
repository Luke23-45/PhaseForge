"""Subprocess execution for runner steps.

Runs each command with inherited stdio (so progress bars and per-epoch logs
stream to the console) and raises on a non-zero exit. The console entry
points are resolved via ``shutil.which`` so the sweep works regardless of
the active environment, with a loud error if they are not installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from phaseforge.runner.commands import eval_command, train_command
from phaseforge.runner.protocol import Step


class CommandError(RuntimeError):
    """Raised when a runner step's command cannot run or fails."""


def _resolve_script(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise CommandError(
            f"Cannot find the {name!r} entry point on PATH. Run "
            "'uv sync' (or reinstall the package) so the console scripts "
            "are available, then re-run the sweep."
        )
    return found


def step_command(
    step: Step, *, ckpt_path: Path | None, outputs_base: Path, defaults: tuple[str, ...]
) -> list[str]:
    """Return the argv for a step (train or eval)."""
    if step.kind == "eval":
        if ckpt_path is None:
            raise CommandError(f"Eval step {step.label} has no checkpoint to evaluate.")
        return eval_command(step, ckpt_path=ckpt_path, outputs_base=outputs_base, defaults=defaults)
    return train_command(step, outputs_base=outputs_base, defaults=defaults, ckpt_path=ckpt_path)


def run_step(
    step: Step,
    *,
    ckpt_path: Path | None,
    outputs_base: Path,
    defaults: tuple[str, ...],
    cwd: Path,
    log_level: str = "WARNING",
) -> None:
    """Execute one step's command, raising :class:`CommandError` on failure.

    ``ckpt_path`` is required for eval steps. ``log_level`` is forwarded as
    ``project.log_level`` so sweep output stays readable unless ``--verbose``
    is requested.
    """
    if step.kind == "eval" and ckpt_path is None:
        raise CommandError(f"Eval step {step.label} has no checkpoint to evaluate.")
    cmd = step_command(step, ckpt_path=ckpt_path, outputs_base=outputs_base, defaults=defaults)
    executable = _resolve_script(cmd[0])
    argv = [executable, f"project.log_level={log_level}"] + cmd[1:]

    print(f"\n[runner] $ {' '.join(argv)}", flush=True)
    result = subprocess.run(argv, cwd=str(cwd))
    if result.returncode != 0:
        raise CommandError(
            f"Step {step.label} failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(argv)}"
        )
