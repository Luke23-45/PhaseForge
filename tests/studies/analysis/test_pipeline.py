"""Tests for the studies.analysis publication pipeline.

The fixture builds a synthetic mini-sweep (two namespaces with a handful of
completed cells) that satisfies the real manifests structurally — the
registry/coverage logic runs against the true manifests, and loaders run
against fabricated artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from studies.analysis.common.config import get_config  # noqa: E402
from studies.analysis.dataset import build_dataset  # noqa: E402


@pytest.fixture()
def mini_sweep(tmp_path: Path, monkeypatch) -> Path:
    """Fabricate outputs_final/ + outputs_ablation/ covering every expected cell.

    Coverage uses the REAL manifests (10x5x3 matrix + 27x3 ablation), so the
    fixture fabricates all 150 + 81 eval cells and their stage runs — files
    are tiny, so this is fast.
    """
    from studies.analysis.common import registry

    final_root = tmp_path / "outputs_final"
    ablation_root = tmp_path / "outputs_ablation"

    def make_train(
        namespace: str,
        root: Path,
        task: str | None,
        method: str,
        seed: int,
        stage: int,
        model_name: str,
        nmi: float = 0.3,
    ) -> None:
        # Mirror the runner's output_tag (task__tag) so sibling variants of one
        # model (bc vs bc_robot_only) get distinct run dirs; ablation cells
        # carry no tag, so disambiguate their run-dir names by method name.
        parts = [task, "robot_only"] if method == "bc_robot_only" else [task]
        tag = "__".join(p for p in parts if p) or None
        dir_tag = tag if tag else method
        run = (
            root
            / model_name
            / f"stage{stage}"
            / f"seed{seed}"
            / f"2026-09-01_00-00-00_{dir_tag}_deadbeef"
        )
        run.mkdir(parents=True, exist_ok=True)
        (run.parent / f"{run.name}.completed").write_text("")
        (run / "run_meta.json").write_text(
            json.dumps(
                {
                    "kind": "train",
                    "model_name": model_name,
                    "stage": stage,
                    "seed": seed,
                    "tag": tag,
                    "method": method,
                    "git_commit": "test0001",
                    "config_hash": "c" * 16,
                    "data_config_hash": "d" * 16,
                }
            )
        )
        rows = []
        for epoch in range(1, 4):
            row = {"epoch": epoch, "global_step": epoch * 10, "val/loss_action": 0.5 / epoch}
            if stage == 2:
                row.update(
                    {
                        "val/phase_expert_nmi": nmi * epoch / 3,
                        "val/routing_entropy": 1.5 - 0.1 * epoch,
                        "val/routing_switch_rate": 0.2,
                        "val/top1_balance_score": 0.5,
                        "val/top1_collapse_rate": 0.1,
                        "train/loss_balance": 0.05,
                        "train/lr": 1e-4,
                        "train_steps_per_second": 90.0,
                        "peak_gpu_memory_mb": 512.0,
                    }
                )
            rows.append(row)
        with (run / "training_curves.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        meta = run / "metadata"
        meta.mkdir(exist_ok=True)
        if stage == 2:
            (meta / "init_routing.json").write_text(
                json.dumps(
                    {
                        "t0_nmi_phase_top1": nmi,
                        "t0_mean_routing_entropy": 1.7,
                        "t0_normalized_routing_entropy": 0.9,
                        "t0_collapse_rate": 0.0,
                        "t0_dead_expert_count": 0,
                        "t0_phase_head_accuracy": 0.6,
                        "t0_top1_expert_frequencies": [0.3, 0.2, 0.2, 0.1, 0.1, 0.1],
                    }
                )
            )
            (meta / "init_expert.json").write_text(
                json.dumps(
                    {
                        "router": {"num_experts": 6, "top_k": 2, "init_type": "centroid"},
                        "expert_init": {
                            "type": "partial_warm",
                            "drop_rate": 0.5,
                            "init_seed": seed,
                            "dropped_indices_sha256": "9113226c" + "0" * 56,
                        },
                        "training_seed": seed,
                    }
                )
            )
        (meta / "environment.json").write_text(
            json.dumps(
                {
                    "git_branch": "v2",
                    "git_sha": "test0001",
                    "platform": "test",
                    "python": "3.11",
                    "extra": {"device": "cpu"},
                    "packages": {"torch": "2.13.0"},
                }
            )
        )
        (run / "timings.json").write_text(json.dumps({"wall_seconds": 12.5}))

    def make_eval(
        namespace: str,
        root: Path,
        task: str | None,
        method: str,
        seed: int,
        model_name: str,
        successes: int = 28,
    ) -> None:
        parts = [task, "robot_only"] if method == "bc_robot_only" else [task]
        tag = "__".join(p for p in parts if p) or None
        dir_tag = tag if tag else method
        run = root / "eval" / model_name / f"seed{seed}" / f"2026-09-01_00-00-01_{dir_tag}_cafe1234"
        run.mkdir(parents=True, exist_ok=True)
        (run.parent / f"{run.name}.completed").write_text("")
        (run / "run_meta.json").write_text(
            json.dumps(
                {
                    "kind": "eval",
                    "model_name": model_name,
                    "seed": seed,
                    "tag": tag,
                    "method": method,
                    "git_commit": "test0001",
                }
            )
        )
        lo, hi = 0.42, 0.68  # overwritten below with the true interval
        from studies.analysis.stats.intervals import wilson_interval

        lo, hi = wilson_interval(successes, 50)
        (run / "eval_results.json").write_text(
            json.dumps(
                {
                    "eval/action_mse": 0.01,
                    "eval/rollout/success_rate": successes / 50,
                    "eval/rollout/valid_episodes": 50,
                    "eval/rollout/successes": successes,
                    "eval/rollout/policy_failures": 0,
                    "eval/rollout/invalid_attempts": 0,
                    "eval/rollout/wilson_ci95_low": lo,
                    "eval/rollout/wilson_ci95_high": hi,
                    "eval/rollout/horizon": 500,
                    "eval/rollout/reset_bank": "bank0001",
                }
            )
        )
        (run / "rollout_summary.json").write_text(
            json.dumps(
                {
                    "router_mode": "learned",
                    "checkpoint_sha256": "c" * 64,
                    "reset_seed": 2026,
                    "reset_bank": "bank0001",
                    "failure_categories": {"task_timeout": 50 - successes},
                }
            )
        )
        with (run / "episodes.jsonl").open("w", encoding="utf-8") as f:
            for idx in range(50):
                success = idx < successes
                f.write(
                    json.dumps(
                        {
                            "episode_index": idx,
                            "success": success,
                            "valid_episode": True,
                            "steps": 60 if success else 500,
                            "timed_out": not success,
                            "termination_reason": "success" if success else "task_timeout",
                            "extra": {"max_phase": (idx % 6)},
                            "task": task,
                            "training_seed": seed,
                            "run_id": "cafe1234",
                        }
                    )
                    + "\n"
                )

    for method in registry.methods("final"):
        for seed in registry.seeds("final"):
            for stage in method.stages:
                make_train(
                    "final", final_root, method.task, method.name, seed, stage, method.model_name
                )
            make_eval(
                "final",
                final_root,
                method.task,
                method.name,
                seed,
                method.model_name,
                successes=25 + (seed % 3),
            )
    for method in registry.methods("ablation"):
        for seed in registry.seeds("ablation"):
            for stage in method.stages:
                make_train(
                    "ablation",
                    ablation_root,
                    None,
                    method.name,
                    seed,
                    stage,
                    method.model_name,
                    nmi=0.4,
                )
            make_eval(
                "ablation",
                ablation_root,
                None,
                method.name,
                seed,
                method.model_name,
                successes=20 + (seed % 5),
            )

    paper = tmp_path / "paper"
    config = tmp_path / "analysis.yaml"
    config.write_text(
        "namespaces:\n"
        "  final:\n"
        f"    root: {final_root.as_posix()}\n"
        "    manifest: experiments/five_task.json\n"
        "  ablation:\n"
        f"    root: {ablation_root.as_posix()}\n"
        "    manifest: experiments/lift_ablation.json\n"
        "output:\n"
        f"  paper_root: {paper.as_posix()}\n"
        "  manifest_name: generation_manifest.json\n"
        "style:\n"
        "  palette: okabe_ito\n"
        "  accent: vermillion\n"
        "  text_width_in: 5.5\n"
        "  margin_width_in: 2.25\n"
        "  dpi: 300\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PHASEFORGE_ANALYSIS_CONFIG", str(config))
    get_config.cache_clear()
    yield tmp_path
    get_config.cache_clear()


@pytest.fixture()
def dataset(mini_sweep: Path):
    return build_dataset(strict=True)


def test_coverage_complete(dataset) -> None:
    report = dataset.coverage()
    assert report.ok, report.summary()
    assert report.present_evals == 150 + 81


def test_loaders_typed_and_fail_closed(dataset, mini_sweep: Path) -> None:
    ev = dataset.matrix_eval("Lift", "phaseforge", 42)
    assert ev.reset_bank == "bank0001"
    assert ev.router_mode == "learned"
    assert ev.failure_categories["task_timeout"] >= 0
    curve = dataset.curve("Lift", "phaseforge", 42, 2)
    assert curve.series("nmi")[-1][1] > curve.series("nmi")[0][1]
    assert curve.series("val/phase_expert_nmi") == curve.series("nmi")
    episodes = dataset.episodes[("Lift", "phaseforge", 42)]
    assert len(episodes) == 50 and episodes[0].max_phase is not None

    from studies.analysis.loaders.eval_results import load_eval_result

    bad_eval_dir = mini_sweep / "outputs_final" / "eval" / "phaseforge" / "seed42"
    target = next(bad_eval_dir.iterdir())
    (target / "eval_results.json").write_text('{"nope": 1}')
    with pytest.raises(ValueError, match="missing required keys"):
        load_eval_result(target)


def test_paired_deltas_and_holm(dataset) -> None:
    from studies.analysis.stats.multiplicity import holm_adjust
    from studies.analysis.stats.paired import pair_episodes

    outcome = pair_episodes(
        "Lift",
        42,
        dataset.episodes[("Lift", "phaseforge", 42)],
        dataset.episodes[("Lift", "bc", 42)],
    )
    assert outcome.n_cases == 50
    assert abs(outcome.delta - (0.25 + 0)) < 0.35  # fabricated cells differ by seed offsets
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_wilson_matches_stored_intervals(dataset) -> None:
    from studies.analysis.stats.intervals import wilson_interval

    # Cross-check EVERY stored interval against the recomputation.
    mismatches = []
    for key, ev in dataset.evals.items():
        lo, hi = wilson_interval(ev.successes, ev.valid_episodes)
        if abs(lo - ev.wilson_low) > 1e-9 or abs(hi - ev.wilson_high) > 1e-9:
            mismatches.append(key)
    assert not mismatches, mismatches[:5]


def test_registry_matches_plan() -> None:
    from studies.analysis.assets import ASSET_REGISTRY, load_generator
    from studies.analysis.common import registry

    assert len(ASSET_REGISTRY) == 23
    assert ASSET_REGISTRY["F1"].kind == "schematic"
    for spec in ASSET_REGISTRY.values():
        if spec.kind != "schematic":
            assert callable(load_generator(spec)), spec.id
    assert len(registry.matrix_method_names()) == 10
    assert len(registry.ablation_method_names()) == 18


def test_generate_and_verify_end_to_end(dataset, mini_sweep: Path, capsys) -> None:
    from studies.analysis.scripts import generate as generate_cli
    from studies.analysis.scripts import verify as verify_cli

    assert generate_cli.main(["--allow-partial"]) == 0
    paper = mini_sweep / "paper"
    assert (paper / "tables" / "T1_success_matrix.tex").is_file()
    assert (paper / "tables" / "T1_success_matrix.md").is_file()
    assert (paper / "figures" / "main" / "F2_paired_deltas.pdf").is_file()
    assert (paper / "figures" / "appendix" / "A14_phase_depth.png").is_file()
    manifest = json.loads((mini_sweep / "generation_manifest.json").read_text())
    assert "T1" in manifest["assets"]

    # verify fails only on the manual schematic (F1) missing.
    assert verify_cli.main([]) == 1
    f1 = paper / "figures" / "main" / "F1_overview.pdf"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"%PDF-1.4 minimal schematic\n")
    assert verify_cli.main([]) == 0


def test_check_mode_reports_coverage(mini_sweep: Path, capsys) -> None:
    from studies.analysis.scripts import generate as generate_cli

    assert generate_cli.main(["--check"]) == 0
