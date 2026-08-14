"""Backfill ``tag``/``method`` identity fields into legacy result ledgers.

Legacy ``results.jsonl`` and ``training_summary.jsonl`` rows (written before
the schema gained the ``tag``/``method`` fields) only record ``model``, so
data-variant runs that share a model name — e.g. the ``bc`` floor and the
``bc``/``robot_only`` negative control — were merged by the summarizers.

This one-shot migration reconstructs the missing identity from each run's
``run_meta.json`` (which recorded ``tag`` all along). Ledgers are rewritten
in place, preserving row order; every modified row is re-validated. Rows that
already carry ``tag`` (post-fix runs) are left untouched.

Usage:
    uv run python scripts/backfill_tags.py [--outputs outputs]

Idempotent: re-running it after the backfill is a no-op.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phaseforge.outputs_writer.backfill import (
    backfill_results,
    backfill_training_summary,
    collect_run_meta,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
        help="Outputs base directory (default: %(default)s).",
    )
    args = parser.parse_args()

    outputs_base = args.outputs.resolve()
    results_dir = outputs_base / "_results"
    index = collect_run_meta(outputs_base)
    print(f"Indexed {len(index)} run_meta.json files under {outputs_base}")

    for ledger, fn in (
        ("results.jsonl", backfill_results),
        ("training_summary.jsonl", backfill_training_summary),
    ):
        if not (results_dir / ledger).exists():
            continue
        report = fn(results_dir, index)
        print(f"{ledger}: {report['changed']}/{report['rows']} rows backfilled")


if __name__ == "__main__":
    main()
