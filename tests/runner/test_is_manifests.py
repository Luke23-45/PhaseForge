"""CPU-only tests for the IS-PhaseForge sweep manifests (WP9).

Every manifest must load under the frozen protocol validator and every
cell must compose under Hydra (catches struct-mode key errors and bad
group/model names without running anything).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize

from phaseforge.runner.protocol import build_plan, load_protocol

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "experiments"

IS_MANIFESTS = (
    "is_stageA_repro.json",
    "is_stageB_discovery.json",
    "is_stageC_routing.json",
    "is_stageD_impedance.json",
    "is_stageE_ablations.json",
    "is_stageF_confirm.json",
    "is_phaseforge_matrix.json",
)


@pytest.mark.parametrize("name", IS_MANIFESTS)
def test_is_manifest_loads(name: str) -> None:
    protocol = load_protocol(EXPERIMENTS / name)
    assert protocol.methods
    assert set(protocol.seeds) <= {42, 43, 44}
    assert all(m.task == "Can" for m in protocol.methods)


def test_is_manifest_method_counts() -> None:
    expected = {
        "is_stageA_repro.json": 2,
        "is_stageB_discovery.json": 5,
        "is_stageC_routing.json": 6,
        "is_stageD_impedance.json": 5,
        "is_stageE_ablations.json": 9,
        "is_stageF_confirm.json": 3,
        "is_phaseforge_matrix.json": 5,
    }
    for name, count in expected.items():
        protocol = load_protocol(EXPERIMENTS / name)
        assert len(protocol.methods) == count, name


@pytest.mark.parametrize("name", IS_MANIFESTS)
def test_is_manifest_cells_compose(name: str) -> None:
    """Every cell composes exactly as the sweep runner invokes it.

    Train steps compose ``train=stageN`` with the default eval group
    (mirrors ``train_command``); eval steps additionally select the eval
    group matching ``evaluate_mode`` (mirrors ``eval_command``). This
    catches struct-mode key errors in method overrides for both shapes.
    """
    protocol = load_protocol(EXPERIMENTS / name)
    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        for method in protocol.methods:
            for stage in method.stages:
                cfg = compose(
                    config_name="main",
                    overrides=[
                        f"models={method.model}",
                        f"data={method.data}",
                        f"train=stage{stage}",
                        *method.overrides,
                    ],
                )
                assert cfg.data.source.task_name == "Can"
            if method.evaluate:
                eval_group = "rollout" if method.evaluate_mode == "rollout" else "metrics"
                cfg = compose(
                    config_name="main",
                    overrides=[
                        f"models={method.model}",
                        f"data={method.data}",
                        f"eval={eval_group}",
                        f"eval.mode={method.evaluate_mode}",
                        *method.overrides,
                    ],
                )
                assert cfg.data.source.task_name == "Can"


@pytest.mark.parametrize("name", IS_MANIFESTS)
def test_is_manifest_builds_plan(name: str) -> None:
    protocol = load_protocol(EXPERIMENTS / name)
    steps = build_plan(protocol, list(protocol.methods), seeds=[42])
    assert steps
    assert all(s.seed == 42 for s in steps)
