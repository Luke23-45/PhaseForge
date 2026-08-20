"""V2 gates G1/G3/G4/G8: cloud experiment launchers + findings collection.

The gate cells already exist in the lift_ablation manifest and run through
``phaseforge-sweep`` (commit-gated registry, auto-injected provider
dependencies, per-method tags). Each gate lives in its own output namespace
so the stepwise gating from report1.md stays enforceable:

* ``g1`` — teacher_forced re-check (EXP-116): re-run the privileged-training
  diagnostic under the current codebase. The old cell predates V2-B; the
  phaseforge stage-1 provider is auto-injected into this namespace.
* ``g3`` — same-wave same-bank re-baseline (EXP-102 + EXP-101): re-run the
  ``bc`` floor and ``phaseforge`` on a fresh wave; ``eval/rollout/reset_bank``
  now records the frozen bank identity on every episode and summary, which
  the Wave-1 cells lacked.
* ``g4`` — warm-vs-reset under V2-B (EXP-105 + EXP-106): re-run
  ``scratch_moe`` (reset) and ``warmstart_moe`` (warm) against the soft
  mapping scheme. Gated on the G2 separability verdict by convention: run
  only after reviewing ``outputs/_findings/phase_merge_separability.json``.
* ``g6`` — E=6 centroid diagnostic (EXP-209): re-run the report1
  phaseforge configuration (``num_experts=6``, ``router_init.type=centroid``)
  on the shared wave-3 bank to separate the bank effect from the V2-B
  config change when the G3 re-baseline disagrees with the Wave-1 number.
* ``g8`` — Wave 3 expert-init (EXP-210 + EXP-211): Drop-Upcycling-style
  partial reinitialization (``warmstart_r50``) and the one-warm-plus-random
  diagnostic (``pf_one_warm_plus_random``). Both pinned to ``num_experts=6``
  + centroid router so the comparison against the g6 baseline (and against
  ``pf_centroid_random``) is expert-init-only.

Per-gate findings (SR + CI + bank id per seed) are collected from each
cell's ``rollout_summary.json`` into ``outputs/_findings/v2_gates_<gate>.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from phaseforge.runner.protocol import load_protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "protocol"))
from smoke_matrix import (  # noqa: E402
    _find_run_by_meta,
)

FINDINGS_DIR = Path("outputs/_findings")

GATES: dict[str, dict] = {
    "g1": {
        "methods": ["teacher_forced"],
        "namespace": "outputs/v2_g1",
        "role": "teacher_forced re-check under V2-B (EXP-116)",
    },
    "g3": {
        "methods": ["bc", "phaseforge"],
        "namespace": "outputs/v2_g3",
        "role": "same-wave same-bank re-baseline (EXP-102 + EXP-101)",
    },
    "g4": {
        "methods": ["scratch_moe", "warmstart_moe"],
        "namespace": "outputs/v2_g4",
        "role": "warm-vs-reset under V2-B (EXP-105 + EXP-106)",
    },
    "g6": {
        "methods": ["phaseforge_e6"],
        "namespace": "outputs/v2_e6_diag",
        "role": "E=6 centroid re-run on shared bank (EXP-209)",
    },
    "g8": {
        "methods": ["warmstart_r50", "pf_one_warm_plus_random"],
        "namespace": "outputs/v2_g8",
        "role": "Wave 3 expert-init: Drop-Upcycling-style partial reinit + "
        "one-warm diagnostic, pinned to E=6 + centroid (EXP-210 + EXP-211)",
        "baseline_gate": "g6",
        "baseline_method": "phaseforge_e6",
    },
}

SWEEP = "phaseforge-sweep"


def _sweep_command(gate: str, seeds: str, manifest: Path, dry_run: bool) -> list[str]:
    spec = GATES[gate]
    cmd = [
        SWEEP,
        "--manifest",
        str(manifest),
        "--methods",
        ",".join(spec["methods"]),
        "--seeds",
        seeds,
        "--outputs",
        spec["namespace"],
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _collect(gate: str, manifest: Path, seeds: list[int], outputs: Path) -> dict:
    protocol = load_protocol(manifest)
    methods = {m.name: m for m in protocol.methods}
    spec = GATES[gate]
    results: dict[str, dict] = {}
    for name in spec["methods"]:
        method = methods[name]
        for seed in seeds:
            try:
                run_dir = _find_run_by_meta(
                    outputs / "eval" / method.model_name / f"seed{seed}",
                    method.name,
                    seed,
                    method.output_tag,
                )
            except RuntimeError as exc:
                results[f"{name}:{seed}"] = {"error": str(exc)}
                continue
            summary = run_dir / "rollout_summary.json"
            if not summary.is_file():
                results[f"{name}:{seed}"] = {
                    "error": f"missing {summary.relative_to(PROJECT_ROOT)}"
                }
                continue
            data = json.loads(summary.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})
            results[f"{name}:{seed}"] = {
                "method": name,
                "seed": seed,
                "sr": metrics.get("eval/rollout/success_rate"),
                "ci_low": metrics.get("eval/rollout/wilson_ci95_low"),
                "ci_high": metrics.get("eval/rollout/wilson_ci95_high"),
                "valid_episodes": metrics.get("eval/rollout/valid_episodes"),
                "reset_bank": metrics.get("eval/rollout/reset_bank"),
                "run_dir": str(run_dir),
            }
    return results


def _verify_bank_against_baseline(
    gate: str,
    seeds: list[int],
    results: dict,
    spec: dict,
) -> tuple[bool, str]:
    """Fail if any g8 rollout was on a different reset bank than the baseline.

    Compares the ``reset_bank`` recorded in each g8 rollout against the same
    seed's baseline (typically g6 ``phaseforge_e6``). Returns ``(True, "")``
    if the baseline findings are missing or if all banks match; returns
    ``(False, error_msg)`` on the first mismatch.

    A gate opt in by setting ``baseline_gate`` and ``baseline_method`` in its
    GATES entry. Currently only ``g8`` uses this guard, because the Wave 3
    expert-init cells are only meaningful when compared against the g6 E=6
    centroid baseline on the same reset bank.
    """
    baseline_gate = spec.get("baseline_gate")
    baseline_method = spec.get("baseline_method")
    if baseline_gate is None or baseline_method is None:
        return True, ""

    baseline_findings_path = FINDINGS_DIR / f"v2_gates_{baseline_gate}.json"
    if not baseline_findings_path.is_file():
        print(
            f"[v2-gates] {gate}: bank-verification skipped — baseline "
            f"findings not found at {baseline_findings_path}. Run "
            f"`--gates {baseline_gate}` first to record the baseline banks."
        )
        return True, ""

    baseline = json.loads(baseline_findings_path.read_text(encoding="utf-8"))
    baseline_results = baseline.get("results", {})

    for name in spec["methods"]:
        for seed in seeds:
            cell = results.get(f"{name}:{seed}", {})
            if "error" in cell:
                continue
            cell_bank = cell.get("reset_bank")
            baseline_key = f"{baseline_method}:{seed}"
            baseline_cell = baseline_results.get(baseline_key, {})
            baseline_bank = baseline_cell.get("reset_bank")
            if baseline_bank is None:
                return False, (
                    f"{gate}: baseline cell {baseline_key} missing "
                    f"reset_bank in {baseline_findings_path.name}"
                )
            if cell_bank != baseline_bank:
                return False, (
                    f"{gate}: bank mismatch for {name}:{seed}: cell "
                    f"reset_bank={cell_bank!r} != baseline "
                    f"{baseline_method}:{seed} reset_bank={baseline_bank!r} "
                    f"(from {baseline_findings_path.name}). The comparison "
                    "is only valid on a matched reset bank — re-run g8 "
                    f"after re-running {baseline_gate} on the same bank."
                )
    print(
        f"[v2-gates] {gate}: bank-verification passed against "
        f"{baseline_method} (baseline gate {baseline_gate})."
    )
    return True, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gates",
        nargs="+",
        default=[],
        help="Gates to run: g1, g3, g4, g6, g8 (default: all).",
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--manifest", default=PROJECT_ROOT / "experiments" / "lift_ablation.json")
    parser.add_argument("--outputs-root", default=PROJECT_ROOT)
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Skip the sweep; only collect findings from existing runs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    gates = [g for g in args.gates if g in GATES]
    unknown = set(args.gates) - set(GATES)
    if unknown:
        print(f"v2_gates: unknown gate(s): {sorted(unknown)}", file=sys.stderr)
        return 2
    if not gates:
        gates = list(GATES)

    if (
        not args.collect_only
        and "g4" in gates
        and not (FINDINGS_DIR / "phase_merge_separability.json").is_file()
    ):
        print(
            "v2_gates: g4 is gated on the G2 separability verdict — "
            "run scripts/experiments/phase_merge_separability.py first",
            file=sys.stderr,
        )
        return 2

    manifest = Path(args.manifest)
    outputs_root = Path(args.outputs_root)
    overall_rc = 0
    for gate in gates:
        spec = GATES[gate]
        print(f"\n[v2-gates] {gate}: {spec['role']}")
        if not args.collect_only:
            cmd = _sweep_command(gate, args.seeds, manifest, args.dry_run)
            print(f"[v2-gates] {gate}: {' '.join(cmd)}")
            if args.dry_run:
                continue
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if proc.returncode != 0:
                print(f"[v2-gates] {gate}: sweep rc={proc.returncode}", file=sys.stderr)
                overall_rc = proc.returncode
                continue
        seeds = [int(s) for s in args.seeds.split(",")]
        results = _collect(gate, manifest, seeds, (outputs_root / spec["namespace"]).resolve())
        bank_ok, bank_err = _verify_bank_against_baseline(gate, seeds, results, spec)
        if not bank_ok:
            print(f"[v2-gates] {gate}: BANK CHECK FAILED — {bank_err}", file=sys.stderr)
            overall_rc = 2
            continue
        payload = {
            "gate": gate,
            "role": spec["role"],
            "methods": spec["methods"],
            "seeds": [int(s) for s in args.seeds.split(",")],
            "results": results,
        }
        FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
        out = FINDINGS_DIR / f"v2_gates_{gate}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[v2-gates] {gate}: findings -> {out}")
        for key, cell in sorted(results.items()):
            if "error" in cell:
                print(f"  {key}: ERROR {cell['error']}")
            else:
                print(
                    f"  {key}: SR={cell['sr']} "
                    f"[{cell['ci_low']},{cell['ci_high']}] "
                    f"bank={cell['reset_bank']}"
                )
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
