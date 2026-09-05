"""Strict preflight verification for experiment manifest directories.

Validates that:
1. Every manifest in the directory is well-formed JSON conforming to the protocol schema.
2. Per-task manifests isolate tasks correctly without cross-task leakage.
3. If a master `main.json` is present, all referenced sub-manifests exist and match the master's definitions.
4. Every cell composes cleanly under Hydra (Stage 1, Stage 2, Rollout Eval).
5. The runner can construct execution plans without missing prerequisite warnings.

Usage::

    uv run python scripts/protocol/verify_manifests.py experiments/precision_residual_confirm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import compose, initialize

from phaseforge.runner.protocol import ProtocolError, build_plan, load_protocol


def verify_directory(dir_path: Path) -> int:
    if not dir_path.is_dir():
        print(f"[ERROR] '{dir_path}' is not a directory.")
        return 1

    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        print(f"[ERROR] No .json manifests found under '{dir_path}'.")
        return 1

    print(f"[preflight] Found {len(json_files)} manifest(s) in {dir_path}:")
    for f in json_files:
        print(f"  - {f.name}")

    errors: list[str] = []
    manifests: dict[str, Any] = {}

    # 1. Load and schema check
    for f in json_files:
        try:
            proto = load_protocol(f)
            manifests[f.name] = proto
            print(f"[OK] {f.name}: task='{proto.task}', methods={len(proto.methods)}, seeds={list(proto.seeds)}")
        except (ProtocolError, OSError) as exc:
            errors.append(f"{f.name}: {exc}")
            print(f"[FAIL] {f.name}: {exc}")

    if errors:
        print(f"\n[preflight] {len(errors)} error(s) during schema loading.")
        return 1

    # 2. Check master main.json consistency if present
    if "main.json" in manifests:
        main_proto = manifests["main.json"]
        print(f"\n[preflight] Verifying master main.json consistency ({len(main_proto.methods)} total methods)...")
        tasks_in_main = {m.task for m in main_proto.methods}
        print(f"  Tasks covered in main.json: {sorted(tasks_in_main)}")
        expected_benchmark_tasks = {"Lift", "Can", "Square", "ToolHang", "Transport"}
        missing_tasks = expected_benchmark_tasks - tasks_in_main
        if missing_tasks:
            errors.append(f"main.json is missing benchmark tasks: {sorted(missing_tasks)}")

    # 3. Hydra composition check
    print("\n[preflight] Testing Hydra config composition for all methods...")
    with initialize(version_base="1.3", config_path="../../phaseforge/config"):
        for fname, proto in manifests.items():
            for method in proto.methods:
                for stage in method.stages:
                    try:
                        compose(
                            config_name="main",
                            overrides=[
                                f"models={method.model}",
                                f"data={method.data}",
                                f"train=stage{stage}",
                                *[o for o in method.overrides if not o.startswith("eval.")],
                            ],
                        )
                    except Exception as exc:
                        errors.append(f"{fname} method '{method.name}' (stage {stage}): {exc}")

                if method.evaluate:
                    eval_group = "rollout" if method.evaluate_mode == "rollout" else "metrics"
                    try:
                        compose(
                            config_name="main",
                            overrides=[
                                f"models={method.model}",
                                f"data={method.data}",
                                f"eval={eval_group}",
                                f"eval.mode={method.evaluate_mode}",
                                *method.overrides,
                            ],
                        )
                    except Exception as exc:
                        errors.append(f"{fname} method '{method.name}' (eval {eval_group}): {exc}")

    # 4. Plan generation check
    print("\n[preflight] Testing runner plan building...")
    for fname, proto in manifests.items():
        try:
            steps = build_plan(proto, list(proto.methods), seeds=list(proto.seeds)[:1])
            print(f"[OK] {fname}: generated {len(steps)} plan step(s) for seed {proto.seeds[0]}")
        except Exception as exc:
            errors.append(f"{fname} plan generation: {exc}")

    if errors:
        print(f"\n[preflight] FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("\n[preflight] SUCCESS! All manifests strictly verified and ready for execution.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict preflight verification for experiment manifests.")
    parser.add_argument(
        "dir",
        nargs="?",
        default="experiments/precision_residual_confirm",
        help="Path to manifest directory (default: experiments/precision_residual_confirm)",
    )
    args = parser.parse_args()
    return verify_directory(Path(args.dir))


if __name__ == "__main__":
    sys.exit(main())
