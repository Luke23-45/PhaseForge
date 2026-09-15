"""Contract tests for the one-seed Square regression repair protocol."""

from __future__ import annotations

from pathlib import Path

from hydra import compose
from hydra.initialize import initialize_config_dir

from phaseforge.runner.protocol import build_plan, load_protocol
from phaseforge.runner.verify import verify_command_contract, verify_plan_ordering


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "experiments" / "square_regression_repairs.json"
CONFIG_DIR = (REPO / "phaseforge" / "config").resolve()


def test_protocol_is_square_only_and_one_factor() -> None:
    protocol = load_protocol(MANIFEST)
    assert protocol.known_tasks == ("Square",)
    assert protocol.seeds == (42, 43, 44)
    assert len(protocol.methods) == 5

    rows = {method.name: method for method in protocol.methods}
    assert rows["precision_residual_phaseforge"].stages == (1,)
    assert rows["square_phase_ce_on"].stage2_source == "self"
    assert "train.supcon.zero_ce=false" in rows["square_phase_ce_on"].overrides
    assert "models.router.top_k=2" in rows["square_topology_top2"].overrides
    assert "models.expert.beta=0.1" in rows["square_residual_beta_01"].overrides


def test_protocol_plan_and_hydra_composition() -> None:
    protocol = load_protocol(MANIFEST)
    plan = build_plan(protocol, list(protocol.methods), seeds=[42], with_dependencies=True)
    assert len(plan) == 10
    assert verify_plan_ordering(protocol, plan) == []
    assert verify_command_contract(protocol, plan, REPO / "outputs_square_regression_repairs_seed42") == []

    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        for method in protocol.methods:
            train_overrides = [x for x in method.overrides if not x.startswith("eval.")]
            for stage in method.stages:
                cfg = compose(
                    config_name="main",
                    overrides=[
                        f"models={method.model}",
                        f"data={method.data}",
                        f"train=stage{stage}",
                    ]
                    + train_overrides,
                )
                if method.name == "square_phase_ce_on" and stage == 1:
                    assert cfg.train.supcon.zero_ce is False
                if method.name == "square_topology_top2" and stage == 2:
                    assert cfg.models.router.top_k == 2
                if method.name == "square_residual_beta_01" and stage == 2:
                    assert float(cfg.models.expert.beta) == 0.1

