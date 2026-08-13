"""CPU-only smoke tests for the actual `scripts/run_multi_seed_eval.py`.

These tests invoke the real script as a subprocess (no mocking of the
aggregation path). They do NOT launch eval subprocesses — no LIBERO,
robosuite, GPU, or checkpoints are needed: the ``--dry-run`` path stops
before any ``phaseforge-eval`` call but still exercises argparse, cell/
suite/seed validation, checkpoint resolution, and the exit-code contract.

Exit-code contract under test:
- ``--dry-run`` always exits 0 (nothing was executed);
- unknown cell -> argparse error (exit 2);
- ``--explicit-checkpoint`` pointing at a missing file -> exit 2;
- no tracebacks ever leak to stderr on these paths.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from phaseforge.evaluations.runners.multi_seed_summary import SeedResult

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_multi_seed_eval.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_multi_seed_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=120,
        check=False,
    )


def test_dry_run_exits_zero_no_traceback(tmp_path: Path) -> None:
    proc = run_script(
        "--dry-run",
        "--cells", "bc",
        "--suites", "libero_10",
        "--checkpoint-root", str(tmp_path / "no_checkpoints_here"),
    )
    assert proc.returncode == 0
    assert "dry run" in proc.stdout
    assert "Traceback" not in proc.stderr


def test_unknown_cell_exits_2(tmp_path: Path) -> None:
    proc = run_script(
        "--dry-run",
        "--cells", "not_a_cell",
        "--checkpoint-root", str(tmp_path),
    )
    assert proc.returncode == 2  # argparse error
    assert "unknown cell(s)" in proc.stderr


def test_missing_explicit_checkpoint_exits_2(tmp_path: Path) -> None:
    proc = run_script(
        "--dry-run",
        "--cells", "bc",
        "--explicit-checkpoint", str(tmp_path / "nope.pt"),
    )
    assert proc.returncode == 2
    assert "--explicit-checkpoint not found" in proc.stderr


def test_invalid_suite_exits_2(tmp_path: Path) -> None:
    proc = run_script(
        "--dry-run",
        "--cells", "bc",
        "--suites", "libero_spatial",  # dropped by Decision 2
        "--checkpoint-root", str(tmp_path),
    )
    assert proc.returncode == 2
    assert "invalid choice" in proc.stderr


def test_end_to_end_shared_checkpoint_excludes_comparisons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--explicit-checkpoint: every cell ran the IDENTICAL checkpoint, so the
    payload records ``checkpoint_shared_across_seeds: true`` and head-to-head
    statistics are excluded (review item 9). Aggregation and episode-weighted
    overall still run against the real code."""
    runner = load_runner()
    fake_ckpt = tmp_path / "ckpt.pt"
    fake_ckpt.write_bytes(b"fake")
    out_path = tmp_path / "final_results.json"
    eval_root = tmp_path / "eval"

    def fake_run(cell, model_cfg, suite, seed, checkpoint, eval_root_arg, args):
        rate = {"bc": 0.4, "phaseforge": 0.6}[cell] if suite.name == "libero_90" else 0.2
        return SeedResult(
            seed=seed,
            suite=suite.name,
            success_rate=rate,
            n_episodes_run=suite.n_episodes,
            per_task_rates=[rate] * suite.n_tasks,
            raw_path=None,
        )

    monkeypatch.setattr(runner, "run_single_seed", fake_run)
    monkeypatch.setattr(
        runner, "find_latest_checkpoint", lambda *a, **k: fake_ckpt
    )
    monkeypatch.setattr(
        sys, "argv",
        [
            "run_multi_seed_eval.py",
            "--cells", "bc", "phaseforge",
            "--suites", "libero_90", "libero_10",
            "--explicit-checkpoint", str(fake_ckpt),
            "--eval-root", str(eval_root),
            "--output", str(out_path),
        ],
    )

    assert runner.main() == 0  # all cells complete -> exit 0

    payload = json.loads(out_path.read_text())
    assert payload["checkpoint_shared_across_seeds"] is True
    assert payload["checkpoint_path"] == str(fake_ckpt)
    assert payload["comparisons"] == []
    assert "comparisons_excluded" in payload
    assert payload["episode_weighted_overall"] is True
    assert payload["protocol_notes"], "protocol caveats must ship with results"

    assert len(payload["cells"]) == 2
    first = payload["cells"][0]
    assert first["cell"] == "bc"
    assert first["suites"]["libero_90"]["suite_role"] == "in-distribution"
    assert first["suites"]["libero_10"]["suite_role"] == "zero-shot"
    assert first["suites"]["libero_90"]["complete"] is True
    assert first["overall"]["seeds_valid"] == 3
    # Episode-weighted overall for bc: (0.4*900 + 0.2*100) / 1000
    assert first["overall"]["weighting"] == "episode-weighted"
    assert first["overall"]["mean"] == pytest.approx((0.4 * 900 + 0.2 * 100) / 1000)
    assert first["overall"]["ci95"] is not None


def test_end_to_end_comparisons_independent_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --explicit-checkpoint, head-to-head statistics ARE computed
    (per-cell checkpoints), on the episode-weighted aggregate."""
    runner = load_runner()
    fake_ckpt = tmp_path / "ckpt.pt"
    fake_ckpt.write_bytes(b"fake")
    out_path = tmp_path / "final_results.json"

    def fake_run(cell, model_cfg, suite, seed, checkpoint, eval_root_arg, args):
        rate = {"bc": 0.4, "phaseforge": 0.6}[cell] if suite.name == "libero_90" else 0.2
        return SeedResult(
            seed=seed,
            suite=suite.name,
            success_rate=rate,
            n_episodes_run=suite.n_episodes,
            per_task_rates=[rate] * suite.n_tasks,
            raw_path=None,
        )

    monkeypatch.setattr(runner, "run_single_seed", fake_run)
    monkeypatch.setattr(
        runner, "find_latest_checkpoint", lambda *a, **k: fake_ckpt
    )
    monkeypatch.setattr(
        sys, "argv",
        [
            "run_multi_seed_eval.py",
            "--cells", "bc", "phaseforge",
            "--suites", "libero_90", "libero_10",
            "--checkpoint-root", str(tmp_path),
            "--eval-root", str(tmp_path / "eval"),
            "--output", str(out_path),
        ],
    )

    assert runner.main() == 0

    payload = json.loads(out_path.read_text())
    assert payload["checkpoint_shared_across_seeds"] is False
    assert payload["checkpoint_path"] is None
    assert "comparisons_excluded" not in payload

    comps = payload["comparisons"]
    assert len(comps) == 1  # one unordered pair: (bc, phaseforge)
    pair = comps[0]
    assert (pair["cell_a"], pair["cell_b"]) == ("bc", "phaseforge")
    assert pair["seeds_used"] == [42, 43, 44]
    # Episode-weighted means: bc = (0.4*900 + 0.2*100)/1000,
    # phaseforge = (0.6*900 + 0.2*100)/1000.
    assert pair["mean_a"] == pytest.approx((0.4 * 900 + 0.2 * 100) / 1000)
    assert pair["mean_b"] == pytest.approx((0.6 * 900 + 0.2 * 100) / 1000)
    assert pair["margin_a_minus_b"] == pytest.approx(-(0.2 * 900) / 1000)
    assert pair["prob_a_over_b"] == pytest.approx(0.0)  # degenerate 0.4 vs 0.6
    assert pair["prob_b_over_a"] == pytest.approx(1.0)
    # Fully separated groups: small p (asymptotic, tie-corrected for the
    # within-group ties; the exact 0.1 only applies to tie-free samples).
    assert 0.0 < pair["mann_whitney_u"]["p"] < 0.1


def test_end_to_end_incomplete_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cell with a failed seed must NOT produce exit 0."""
    runner = load_runner()
    fake_ckpt = tmp_path / "ckpt.pt"
    fake_ckpt.write_bytes(b"fake")
    out_path = tmp_path / "final_results.json"

    def fake_run(cell, model_cfg, suite, seed, checkpoint, eval_root_arg, args):
        if seed == 44:  # seed 44 crashes for EVERY cell
            return SeedResult(
                seed=seed, suite=suite.name, success_rate=None,
                n_episodes_run=None, per_task_rates=[], raw_path=None,
                error="simulated crash",
            )
        return SeedResult(
            seed=seed, suite=suite.name, success_rate=0.5,
            n_episodes_run=suite.n_episodes,
            per_task_rates=[0.5] * suite.n_tasks, raw_path=None,
        )

    monkeypatch.setattr(runner, "run_single_seed", fake_run)
    monkeypatch.setattr(
        runner, "find_latest_checkpoint", lambda *a, **k: fake_ckpt
    )
    monkeypatch.setattr(
        sys, "argv",
        [
            "run_multi_seed_eval.py",
            "--cells", "bc",
            "--suites", "libero_90",
            "--explicit-checkpoint", str(fake_ckpt),
            "--eval-root", str(tmp_path / "eval"),
            "--output", str(out_path),
        ],
    )

    assert runner.main() == 1  # 2/3 valid seeds -> incomplete -> exit 1

    payload = json.loads(out_path.read_text())
    assert payload["cells"][0]["suites"]["libero_90"]["complete"] is False
    assert payload["cells"][0]["suites"]["libero_90"]["seeds_valid"] == 2
    assert "44" in payload["cells"][0]["suites"]["libero_90"]["seed_errors"]
