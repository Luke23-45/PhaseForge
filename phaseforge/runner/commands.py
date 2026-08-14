"""Pure argv builders for the CLI commands a plan step executes.

Kept side-effect free and unit-tested: the runner's correctness hinges on
these exact overrides (model, stage, seed, outputs base, data variant tag,
early-stopping lock, and the evaluated checkpoint path).
"""

from __future__ import annotations

from pathlib import Path

from phaseforge.runner.protocol import Step


def train_command(
    step: Step,
    *,
    outputs_base: Path,
    defaults: tuple[str, ...],
    ckpt_path: Path | None = None,
) -> list[str]:
    """Build ``phaseforge-train`` argv for a training step.

    A stage-2 step that bootstraps from a provider passes that exact Stage 1
    checkpoint as ``train.stage1_ckpt_path``, so the subprocess never falls
    back to its own loose auto-detect (which could select a tagged sibling
    variant that shares the provider's output tree).
    """
    method = step.method
    assert step.stage is not None
    cmd = [
        "phaseforge-train",
        f"models={method.model}",
        f"train=stage{step.stage}",
        f"project.seed={step.seed}",
        f"project.output_dir={outputs_base}",
    ]
    if method.data != "common":
        cmd.append(f"data={method.data}")
    if method.tag:
        cmd.append(f"project.tag={method.tag}")
    if method.name:
        cmd.append(f"project.method={method.name}")
    if ckpt_path is not None:
        cmd.append(f"train.stage1_ckpt_path={ckpt_path}")
    cmd.extend(defaults)
    return cmd


def eval_command(
    step: Step,
    *,
    ckpt_path: Path,
    outputs_base: Path,
    defaults: tuple[str, ...],
) -> list[str]:
    """Build ``phaseforge-eval`` argv for an evaluation step.

    The historical override name is ``train.stage1_ckpt_path`` but it is the
    checkpoint actually evaluated (the method's final-stage artifact).
    """
    method = step.method
    cmd = [
        "phaseforge-eval",
        f"models={method.model}",
        f"project.seed={step.seed}",
        f"project.output_dir={outputs_base}",
        f"train.stage1_ckpt_path={ckpt_path}",
    ]
    if method.data != "common":
        cmd.append(f"data={method.data}")
    if method.tag:
        cmd.append(f"project.tag={method.tag}")
    if method.name:
        cmd.append(f"project.method={method.name}")
    cmd.extend(defaults)
    return cmd
