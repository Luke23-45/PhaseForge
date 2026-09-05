"""Tests for the Precision-Residual PhaseForge experiment manifests across all 5 benchmark datasets.

Verifies that:
1. Each individual task manifest (Lift, Can, Square, ToolHang, Transport) loads cleanly.
2. The master manifest `main.json` and the directory path load cleanly and match expected task specifications.
3. Every cell across all manifests composes successfully under Hydra (Stage 1, Stage 2, Rollout Eval).
4. Sweep execution plans can be generated for all 5 tasks without prerequisite or structural errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize

from phaseforge.runner.protocol import build_plan, load_protocol

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = REPO / "experiments" / "precision_residual_confirm"

TASK_MANIFESTS = (
    ("Lift", "precision_residual_confirm_lift.json"),
    ("Can", "precision_residual_confirm_can.json"),
    ("Square", "precision_residual_confirm_square.json"),
    ("ToolHang", "precision_residual_confirm_tool_hang.json"),
    ("Transport", "precision_residual_confirm_transport.json"),
)


@pytest.mark.parametrize("expected_task,filename", TASK_MANIFESTS)
def test_individual_task_manifest_loads(expected_task: str, filename: str) -> None:
    manifest_path = EXP_DIR / filename
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"

    protocol = load_protocol(manifest_path)
    assert protocol.task == expected_task
    assert protocol.seeds == (42, 43, 44)
    assert len(protocol.methods) == 2

    # Method 1: proposed precision_residual_phaseforge
    m1 = protocol.methods[0]
    assert m1.name == "precision_residual_phaseforge"
    assert m1.model == "precision_residual_phaseforge"
    assert m1.task == expected_task
    assert m1.stages == (1, 2)
    assert m1.evaluate is True
    assert m1.evaluate_mode == "rollout"

    # Method 2: control pf_direct_supcon_margin
    m2 = protocol.methods[1]
    assert m2.name == "pf_direct_supcon_margin"
    assert m2.model == "phaseforge_prototype"
    assert m2.task == expected_task


def test_master_main_json_loads() -> None:
    manifest_path = EXP_DIR / "main.json"
    assert manifest_path.exists()

    protocol = load_protocol(manifest_path)
    assert protocol.task == "all"
    assert protocol.seeds == (42, 43, 44)
    assert len(protocol.methods) == 10

    # Ensure all 5 tasks are covered in main.json
    tasks_present = {m.task for m in protocol.methods}
    assert tasks_present == {"Lift", "Can", "Square", "ToolHang", "Transport"}


def test_directory_path_resolves_main_json() -> None:
    protocol = load_protocol(EXP_DIR)
    assert protocol.name == "precision_residual_confirm_five_task"
    assert len(protocol.methods) == 10


@pytest.mark.parametrize("expected_task,filename", TASK_MANIFESTS)
def test_manifest_cells_compose_hydra(expected_task: str, filename: str) -> None:
    protocol = load_protocol(EXP_DIR / filename)
    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        for method in protocol.methods:
            # Stage 1 composition
            cfg_s1 = compose(
                config_name="main",
                overrides=[
                    f"models={method.model}",
                    f"data={method.data}",
                    "train=stage1",
                    *[o for o in method.overrides if not o.startswith("eval.")],
                ],
            )
            assert cfg_s1.data.source.task_name == expected_task

            # Stage 2 composition
            cfg_s2 = compose(
                config_name="main",
                overrides=[
                    f"models={method.model}",
                    f"data={method.data}",
                    "train=stage2",
                    *[o for o in method.overrides if not o.startswith("eval.")],
                ],
            )
            assert cfg_s2.data.source.task_name == expected_task

            # Rollout Eval composition
            eval_group = "rollout" if method.evaluate_mode == "rollout" else "metrics"
            cfg_eval = compose(
                config_name="main",
                overrides=[
                    f"models={method.model}",
                    f"data={method.data}",
                    f"eval={eval_group}",
                    f"eval.mode={method.evaluate_mode}",
                    *method.overrides,
                ],
            )
            assert cfg_eval.data.source.task_name == expected_task


@pytest.mark.parametrize("expected_task,filename", TASK_MANIFESTS)
def test_manifest_builds_plan(expected_task: str, filename: str) -> None:
    protocol = load_protocol(EXP_DIR / filename)
    steps = build_plan(protocol, list(protocol.methods), seeds=[42])
    # 2 methods * (stage1 + stage2 + eval) = 6 steps per seed
    assert len(steps) == 6
    assert all(s.seed == 42 for s in steps)
    assert all(s.method.task == expected_task for s in steps)
