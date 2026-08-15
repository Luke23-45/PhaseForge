"""Pure argv builders for the CLI commands a plan step executes.

Kept side-effect free and unit-tested: the runner's correctness hinges on
these exact overrides (model, stage, seed, outputs base, data variant tag,
early-stopping lock, and the evaluated checkpoint path).
"""

from __future__ import annotations

from pathlib import Path

from phaseforge.runner.protocol import Method, Step


def _effective_tag(method: Method) -> str | None:
    """Compose the effective output tag.

    The five-task protocol trains the same method name on multiple tasks;
    without a task-prefixed tag, the output trees for, e.g.,
    ``phaseforge/Lift`` and ``phaseforge/Can`` would collide on the same
    directory. The task tag is appended to any explicit ``method.tag``
    the manifest carries, so the negative control's ``robot_only`` tag
    survives (for example ``Lift__robot_only``). The separator is
    filesystem-safe because the tag is embedded in one run-directory name.
    """
    return method.output_tag


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
    effective_tag = _effective_tag(method)
    if effective_tag:
        cmd.append(f"project.tag={effective_tag}")
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
    checkpoint actually evaluated (the method's final-stage artifact). The
    evaluation mode comes from the method's ``evaluate_mode`` (rollout by
    default), so the runner never evaluates the state-only protocol with
    the offline metric.
    """
    method = step.method
    cmd = [
        "phaseforge-eval",
        f"models={method.model}",
        f"project.seed={step.seed}",
        f"project.output_dir={outputs_base}",
        f"train.stage1_ckpt_path={ckpt_path}",
        f"eval.mode={method.evaluate_mode}",
    ]
    if method.data != "common":
        cmd.append(f"data={method.data}")
    effective_tag = _effective_tag(method)
    if effective_tag:
        cmd.append(f"project.tag={effective_tag}")
    if method.name:
        cmd.append(f"project.method={method.name}")
    cmd.extend(defaults)
    return cmd
