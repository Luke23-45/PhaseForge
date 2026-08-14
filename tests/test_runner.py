"""CPU-only tests for the experiment runner (protocol, plan, registry, resolver, CLI)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phaseforge.runner import cli as runner_cli
from phaseforge.runner.commands import eval_command, train_command
from phaseforge.runner.executor import CommandError
from phaseforge.runner.protocol import (
    Protocol,
    ProtocolError,
    Step,
    build_plan,
    load_protocol,
)
from phaseforge.runner.registry import RegistryError, RunnerState
from phaseforge.runner.resolver import (
    CheckpointError,
    checkpoint_exists,
    resolve_checkpoint_path,
    resolve_run_dir,
    stage_checkpoint_relative,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "experiments" / "lift_pilot.json"


def _protocol() -> Protocol:
    return load_protocol(MANIFEST)


def _step(method_name: str, seed: int, phase: str) -> Step:
    protocol = _protocol()
    method = protocol.method_by_name(method_name)
    assert method is not None
    if phase == "eval":
        return Step(kind="eval", method=method, seed=seed)
    return Step(kind="train", method=method, seed=seed, stage=int(phase[-1]))


# ---------------------------------------------------------------------------
# Protocol loading / validation
# ---------------------------------------------------------------------------


def test_load_real_protocol_matches_notebook_methods() -> None:
    protocol = _protocol()
    assert protocol.name == "lift_pilot"
    assert protocol.seeds == (42, 43, 44)
    assert [m.index for m in protocol.methods] == list(range(1, 10))
    names = {m.name for m in protocol.methods}
    assert names == {
        "phaseforge",
        "bc",
        "bc_robot_only",
        "scratch_moe",
        "warmstart_moe",
        "phase_pretrain_random_router",
        "plain_encoder_phase_bootstrap",
        "teacher_forced",
        "oracle_moe",
    }
    phaseforge = protocol.method_by_name("phaseforge")
    assert phaseforge is not None
    assert phaseforge.stages == (1, 2)
    assert phaseforge.stage2_source == "self"
    assert phaseforge.evaluate
    assert phaseforge.model_name == "phaseforge"

    bc = protocol.method_by_name("bc")
    assert bc is not None
    assert bc.stages == (1,)
    assert bc.stage2_source is None

    robot = protocol.method_by_name("bc_robot_only")
    assert robot is not None
    assert robot.tag == "robot_only"
    assert robot.model_name == "bc"

    warmstart = protocol.method_by_name("warmstart_moe")
    assert warmstart is not None
    assert warmstart.stage2_source == "bc"

    teacher = protocol.method_by_name("teacher_forced")
    assert teacher is not None
    assert teacher.stage2_source == "phaseforge"


def _write_protocol(tmp_path: Path, doc: dict) -> Path:
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _valid_doc() -> dict:
    return {
        "name": "mini",
        "task": "Mini",
        "seeds": [42],
        "defaults": ["train.early_stopping.enabled=false"],
        "methods": [
            {
                "index": 1,
                "name": "phaseforge",
                "role": "proposed",
                "model": "phaseforge",
                "data": "common",
                "stages": [1, 2],
                "stage2_source": "self",
                "evaluate": True,
            },
            {
                "index": 2,
                "name": "bc",
                "role": "floor",
                "model": "baselines/bc",
                "data": "common",
                "stages": [1],
                "evaluate": True,
            },
        ],
    }


def test_load_protocol_valid(tmp_path: Path) -> None:
    protocol = load_protocol(_write_protocol(tmp_path, _valid_doc()))
    assert len(protocol.methods) == 2
    assert protocol.methods[0].index == 1
    assert protocol.defaults == ("train.early_stopping.enabled=false",)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d["methods"].append(dict(d["methods"][0], index=1)), "Duplicate method index"),
        (lambda d: d.update(seeds=[]), "seeds"),
        (lambda d: d.update(seeds=["x"]), "seed"),
        (lambda d: d["methods"][0].update(stages=[3]), "stage"),
        (lambda d: d["methods"][0].update(stages=[2, 1]), "ascending"),
        (lambda d: d["methods"][0].pop("model"), "model"),
        (lambda d: d["methods"][0].update(index=0), "index"),
        (lambda d: d["methods"][0].update(stage2_source="nope"), "stage2_source"),
        (
            lambda d: d["methods"][0].update(stage2_source="phaseforge", stages=[1]),
            "stage 2",
        ),
    ],
)
def test_load_protocol_validation_errors(tmp_path: Path, mutate, match) -> None:
    doc = _valid_doc()
    mutate(doc)
    with pytest.raises(ProtocolError, match=match):
        load_protocol(_write_protocol(tmp_path, doc))


def test_load_protocol_rejects_unknown_provider(tmp_path: Path) -> None:
    doc = _valid_doc()
    # bc gains a stage 2 sourced from a provider that no longer exists.
    doc["methods"][1].update(stages=[1, 2], stage2_source="phaseforge")
    doc["methods"] = [doc["methods"][1]]  # drop the phaseforge method entirely
    with pytest.raises(ProtocolError, match="not a method"):
        load_protocol(_write_protocol(tmp_path, doc))


def test_select_methods_by_index_and_name() -> None:
    protocol = _protocol()
    selected = protocol.select_methods(["9", "1", "warmstart_moe"])
    assert [m.index for m in selected] == [1, 5, 9]
    with pytest.raises(ProtocolError, match="Unknown method"):
        protocol.select_methods(["does_not_exist"])
    with pytest.raises(ProtocolError, match="Unknown method index"):
        protocol.select_methods(["99"])


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def test_build_plan_full_matrix_order_and_count() -> None:
    protocol = _protocol()
    plan = build_plan(protocol, list(protocol.methods), seeds=[42])
    # phaseforge (3) + 8 single-stage methods (2 each) = 19 steps per seed.
    assert len(plan) == 19
    labels = [s.label for s in plan[:6]]
    assert labels == [
        "phaseforge seed=42 stage1",
        "phaseforge seed=42 stage2",
        "phaseforge seed=42 eval",
        "bc seed=42 stage1",
        "bc seed=42 eval",
        "bc_robot_only seed=42 stage1",
    ]
    # Eval step targets the method's final-stage checkpoint.
    eval_step = plan[2]
    assert eval_step.kind == "eval"
    assert eval_step.required_checkpoint() == ("phaseforge", 2)


def test_build_plan_multi_seed() -> None:
    protocol = _protocol()
    plan = build_plan(protocol, [protocol.method_by_name("bc")], seeds=[42, 44])
    assert [s.seed for s in plan] == [42, 42, 44, 44]


def test_build_plan_stage_filter() -> None:
    protocol = _protocol()
    plan = build_plan(
        protocol, [protocol.method_by_name("phaseforge")], seeds=[42], stage=1
    )
    assert len(plan) == 1
    assert plan[0].stage == 1
    assert plan[0].kind == "train"


def test_build_plan_eval_only() -> None:
    protocol = _protocol()
    plan = build_plan(
        protocol, [protocol.method_by_name("phaseforge")], seeds=[42], eval_only=True
    )
    assert len(plan) == 1
    assert plan[0].kind == "eval"


def test_build_plan_skip_eval() -> None:
    protocol = _protocol()
    plan = build_plan(
        protocol, [protocol.method_by_name("phaseforge")], seeds=[42], skip_eval=True
    )
    assert [s.kind for s in plan] == ["train", "train"]


def test_build_plan_conflicting_filters() -> None:
    protocol = _protocol()
    with pytest.raises(ProtocolError, match="mutually exclusive"):
        build_plan(
            protocol,
            [protocol.method_by_name("bc")],
            seeds=[42],
            stage=1,
            eval_only=True,
        )
    with pytest.raises(ProtocolError, match="mutually exclusive"):
        build_plan(
            protocol,
            [protocol.method_by_name("bc")],
            seeds=[42],
            eval_only=True,
            skip_eval=True,
        )


def test_build_plan_injects_dependency() -> None:
    protocol = _protocol()
    teacher = protocol.method_by_name("teacher_forced")
    plan = build_plan(protocol, [teacher], seeds=[42], with_dependencies=True)
    assert len(plan) == 3
    dep = plan[0]
    assert dep.dependency
    assert dep.method.name == "phaseforge"
    assert dep.stage == 1
    assert dep.registry_phase == "stage1"
    assert plan[1].label == "teacher_forced seed=42 stage2"
    assert plan[2].kind == "eval"


def test_build_plan_no_injection_when_provider_selected() -> None:
    protocol = _protocol()
    selected = [
        protocol.method_by_name("phaseforge"),
        protocol.method_by_name("teacher_forced"),
    ]
    plan = build_plan(protocol, selected, seeds=[42], with_dependencies=True)
    assert not any(s.dependency for s in plan)


def test_build_plan_rejects_unknown_seed() -> None:
    protocol = _protocol()
    with pytest.raises(ProtocolError, match="Seed 99"):
        build_plan(protocol, [protocol.method_by_name("bc")], seeds=[99])


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


def test_train_command_common_cell(tmp_path: Path) -> None:
    cmd = train_command(
        _step("bc", 42, "stage1"),
        outputs_base=tmp_path / "outputs",
        defaults=("train.early_stopping.enabled=false",),
    )
    assert cmd == [
        "phaseforge-train",
        "models=baselines/bc",
        "train=stage1",
        "project.seed=42",
        f"project.output_dir={tmp_path / 'outputs'}",
        "train.early_stopping.enabled=false",
    ]


def test_train_command_variant_tag_and_data(tmp_path: Path) -> None:
    cmd = train_command(
        _step("bc_robot_only", 42, "stage1"),
        outputs_base=tmp_path / "outputs",
        defaults=(),
    )
    assert "data=robot_only" in cmd
    assert "project.tag=robot_only" in cmd


def test_train_command_stage2(tmp_path: Path) -> None:
    cmd = train_command(
        _step("phaseforge", 42, "stage2"),
        outputs_base=tmp_path / "outputs",
        defaults=(),
    )
    assert "models=phaseforge" in cmd
    assert "train=stage2" in cmd


def test_eval_command_targets_final_checkpoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "outputs" / "phaseforge" / "stage2" / "ckpt" / "checkpoint_best.pt"
    cmd = eval_command(
        _step("phaseforge", 42, "eval"),
        ckpt_path=ckpt,
        outputs_base=tmp_path / "outputs",
        defaults=(),
    )
    assert cmd[0] == "phaseforge-eval"
    assert f"train.stage1_ckpt_path={ckpt}" in cmd
    assert "project.seed=42" in cmd


# ---------------------------------------------------------------------------
# Runner state registry
# ---------------------------------------------------------------------------


def test_registry_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "_runner" / "state.json"
    state = RunnerState(path)
    state.mark("phaseforge", 42, "stage1", run_dir="phaseforge/stage1/x", ckpt="p/s1/c.pt")
    state.mark("phaseforge", 42, "stage2", run_dir="phaseforge/stage2/y", ckpt="p/s2/c.pt")
    state.mark("phaseforge", 42, "eval", ckpt="p/s2/c.pt", run_dir="eval/phaseforge/z")

    reloaded = RunnerState(path)
    assert reloaded.is_complete("phaseforge", 42, "stage1")
    assert reloaded.is_complete("phaseforge", 42, "eval")
    assert not reloaded.is_complete("phaseforge", 43, "stage1")
    assert reloaded.get_ckpt("phaseforge", 42, 2) == "p/s2/c.pt"
    assert reloaded.get_ckpt("phaseforge", 43, 1) is None


def test_registry_mark_failed(tmp_path: Path) -> None:
    state = RunnerState(tmp_path / "state.json")
    state.mark_failed("bc", 42, "stage1", "boom")
    assert not state.is_complete("bc", 42, "stage1")
    assert state.get("bc", 42, "stage1")["status"] == "failed"


def test_registry_corrupt_state_is_loud(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(RegistryError, match="corrupt"):
        RunnerState(path)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _make_run(
    base: Path, model: str, stage: int, name: str, seed: int, tag=None,
    completed: bool = True, seed_dir: bool = True,
) -> None:
    run_dir = base / model / f"stage{stage}"
    if seed_dir:
        run_dir = run_dir / f"seed{seed}"
    run_dir = run_dir / name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "checkpoint_best.pt").write_text("dummy")
    meta = {"seed": seed}
    if tag is not None:
        meta["tag"] = tag
    (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if completed:
        # Real runs get the sibling lifecycle marker written by RunWriter
        # only when the run finishes successfully.
        (run_dir.with_name(name + ".completed")).write_text("{}", encoding="utf-8")


def test_resolve_run_dir_seed_and_tag(tmp_path: Path) -> None:
    _make_run(tmp_path, "bc", 1, "2026-08-01_10-00-00_aaaa0001", seed=42)
    _make_run(tmp_path, "bc", 1, "2026-08-02_10-00-00_aaaa0002", seed=42, tag="robot_only")
    _make_run(tmp_path, "bc", 1, "2026-08-03_10-00-00_aaaa0003", seed=43)

    assert resolve_run_dir(tmp_path, "bc", 1, seed=42).name == "2026-08-01_10-00-00_aaaa0001"
    assert (
        resolve_run_dir(tmp_path, "bc", 1, seed=42, tag="robot_only").name
        == "2026-08-02_10-00-00_aaaa0002"
    )
    with pytest.raises(CheckpointError, match="seed 44"):
        resolve_run_dir(tmp_path, "bc", 1, seed=44)


def test_resolve_run_dir_ignores_crashed_runs(tmp_path: Path) -> None:
    # A run killed after its last checkpoint save but before mark_completed
    # has run_meta + checkpoint_best.pt but no .completed marker. It must
    # never be selected as an eval target.
    _make_run(tmp_path, "bc", 1, "2026-08-01_10-00-00_aaaa0001", seed=42,
              completed=False)
    _make_run(tmp_path, "bc", 1, "2026-08-03_10-00-00_aaaa0003", seed=42)

    assert resolve_run_dir(tmp_path, "bc", 1, seed=42).name == "2026-08-03_10-00-00_aaaa0003"
    with pytest.raises(CheckpointError, match="completed"):
        resolve_run_dir(tmp_path, "bc", 1, seed=43)


def test_checkpoint_exists_false_for_crashed_run(tmp_path: Path) -> None:
    _make_run(tmp_path, "bc", 1, "2026-08-01_10-00-00_aaaa0001", seed=42,
              completed=False)
    assert not checkpoint_exists(tmp_path, "bc", 1, seed=42)


def test_resolve_run_dir_missing_stage(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="stage1"):
        resolve_run_dir(tmp_path, "phaseforge", 1, seed=42)


def test_resolve_run_dir_legacy_layout(tmp_path: Path) -> None:
    # Runs written before seeds became a directory dimension sit directly
    # under stage{N}/; the resolver must still find and seed-filter them.
    _make_run(tmp_path, "bc", 1, "2026-08-01_10-00-00_aaaa0001", seed=42,
              seed_dir=False)
    _make_run(tmp_path, "bc", 1, "2026-08-03_10-00-00_aaaa0003", seed=43,
              seed_dir=False)

    assert resolve_run_dir(tmp_path, "bc", 1, seed=42).name == "2026-08-01_10-00-00_aaaa0001"
    assert resolve_run_dir(tmp_path, "bc", 1, seed=43).name == "2026-08-03_10-00-00_aaaa0003"
    with pytest.raises(CheckpointError, match="seed 44"):
        resolve_run_dir(tmp_path, "bc", 1, seed=44)


def test_stage_checkpoint_relative_requires_best_ckpt(tmp_path: Path) -> None:
    run_dir = tmp_path / "bc" / "stage1" / "2026-08-01_10-00-00_aaaa0001"
    run_dir.mkdir(parents=True)
    with pytest.raises(CheckpointError, match="no .*checkpoint_best"):
        stage_checkpoint_relative(tmp_path, run_dir, "bc", 1)


def test_resolve_checkpoint_path_scan_fallback(tmp_path: Path) -> None:
    _make_run(tmp_path, "phaseforge", 2, "2026-08-01_10-00-00_aaaa0001", seed=42)
    state = RunnerState(tmp_path / "state.json")
    method = _protocol().method_by_name("phaseforge")
    assert method is not None
    ckpt = resolve_checkpoint_path(tmp_path, method, 2, seed=42, state=state)
    assert ckpt.is_file()
    assert "aaaa0001" in str(ckpt)

    with pytest.raises(CheckpointError, match="stage2"):
        resolve_checkpoint_path(tmp_path, method, 2, seed=44, state=state)


def test_resolve_checkpoint_path_prefers_registry(tmp_path: Path) -> None:
    _make_run(tmp_path, "phaseforge", 2, "2026-08-01_10-00-00_aaaa0001", seed=42)
    state = RunnerState(tmp_path / "state.json")
    state.mark(
        "phaseforge", 42, "stage2", run_dir="x",
        ckpt="phaseforge/stage2/old/checkpoint_best.pt",
    )
    (tmp_path / "phaseforge" / "stage2" / "old").mkdir(parents=True, exist_ok=True)
    (tmp_path / "phaseforge" / "stage2" / "old" / "checkpoint_best.pt").write_text("newer")
    method = _protocol().method_by_name("phaseforge")
    assert method is not None
    ckpt = resolve_checkpoint_path(tmp_path, method, 2, seed=42, state=state)
    assert "old" in str(ckpt)


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def test_cli_list(tmp_path: Path, capsys) -> None:
    protocol_path = _write_protocol(tmp_path, _valid_doc())
    args = runner_cli.parse_args(["--manifest", str(protocol_path), "--list"])
    assert runner_cli.run(args) == 0
    out = capsys.readouterr().out
    assert "phaseforge" in out
    assert "baselines/bc" in out
    assert "seeds: [42]" in out


def test_cli_dry_run_prints_commands_without_executing(tmp_path: Path, capsys) -> None:
    protocol_path = _write_protocol(tmp_path, _valid_doc())
    outputs = tmp_path / "outputs"
    args = runner_cli.parse_args(
        [
            "--manifest", str(protocol_path),
            "--outputs", str(outputs),
            "--methods", "bc",
            "--seeds", "42",
            "--dry-run",
        ]
    )
    assert runner_cli.run(args) == 0
    out = capsys.readouterr().out
    assert "WOULD RUN  bc seed=42 stage1" in out
    assert "phaseforge-train" in out
    assert "models=baselines/bc" in out
    # Eval cannot run before bc stage1 exists — dry-run reports the blocker.
    assert "BLOCKED  bc seed=42 eval" in out
    assert "No bc stage1 runs found" in out


def test_cli_dry_run_with_state_complete_skips(tmp_path: Path, capsys) -> None:
    protocol_path = _write_protocol(tmp_path, _valid_doc())
    outputs = tmp_path / "outputs"
    state = RunnerState(runner_cli.RunnerState.default_path(outputs))
    state.mark("bc", 42, "stage1", run_dir="bc/stage1/x", ckpt="bc/stage1/x/c.pt")
    args = runner_cli.parse_args(
        [
            "--manifest", str(protocol_path),
            "--outputs", str(outputs),
            "--methods", "bc",
            "--seeds", "42",
            "--dry-run",
        ]
    )
    assert runner_cli.run(args) == 0
    out = capsys.readouterr().out
    assert "skip bc seed=42 stage1 (already completed)" in out


def test_cli_continue_on_error_records_failures(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    protocol_path = _write_protocol(tmp_path, _valid_doc())

    def _boom(*args, **kwargs) -> None:
        raise CommandError("synthetic failure")

    monkeypatch.setattr(runner_cli, "run_step", _boom)
    outputs = tmp_path / "outputs"
    args = runner_cli.parse_args(
        [
            "--manifest", str(protocol_path),
            "--outputs", str(outputs),
            "--methods", "phaseforge",
            "--seeds", "42",
            "--continue-on-error",
        ]
    )
    assert runner_cli.run(args) == 1
    state = RunnerState(runner_cli.RunnerState.default_path(outputs))
    for phase in ("stage1", "stage2", "eval"):
        entry = state.get("phaseforge", 42, phase)
        assert entry is not None and entry["status"] == "failed"
    captured = capsys.readouterr()
    assert "FAILED phaseforge seed=42 stage1" in captured.out + captured.err


def test_cli_fails_fast_without_continue_on_error(tmp_path: Path, monkeypatch) -> None:
    protocol_path = _write_protocol(tmp_path, _valid_doc())

    def _boom(*args, **kwargs) -> None:
        raise CommandError("synthetic failure")

    monkeypatch.setattr(runner_cli, "run_step", _boom)
    args = runner_cli.parse_args(
        [
            "--manifest", str(protocol_path),
            "--outputs", str(tmp_path / "outputs"),
            "--methods", "phaseforge",
            "--seeds", "42",
        ]
    )
    assert runner_cli.run(args) == 1
