"""``phaseforge-gates`` — run the rollout validation gates (§4.5).

Usage::

    phaseforge-gates eval=rollout                 # default experiment config
    phaseforge-gates eval=rollout project.seed=42

Exit codes:
* 0 — every required gate passed (skipped gates are warnings only)
* 1 — one or more required gates FAILED
* 2 — the gates could not run at all (e.g. robosuite missing)

The report is written to ``{outputs}/_gates/{timestamp}/gates_report.json``
under the project output base.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from phaseforge.utils.config import output_base_dir

logger = logging.getLogger(__name__)


def _write_report(cfg: DictConfig, results, exit_code: int, base: Path) -> Path:
    timestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base / "_gates" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": timestamp,
        "exit_code": exit_code,
        "gates": [
            {
                "gate": r.gate,
                "status": r.status,
                "detail": r.detail,
                "metrics": r.metrics,
            }
            for r in results
        ],
    }
    path = out_dir / "gates_report.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@hydra.main(version_base="1.3", config_path="config", config_name="main")
def gates(cfg: DictConfig) -> None:
    """Run all rollout validation gates and report PASS/FAIL/SKIPPED."""
    from phaseforge.evaluations.rollout.gates import GateFailure, run_all_gates

    try:
        results = run_all_gates(cfg)
    except GateFailure as exc:
        logger.error("Gates could not run: %s", exc)
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001 — setup failure = cannot run
        logger.exception("Gates could not run: %s", exc)
        sys.exit(2)

    exit_code = 0
    for result in results:
        marker = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIPPED": "[SKIP]"}[result.status]
        print(f"{marker} {result.gate}: {result.detail}")
        if result.status == "FAIL":
            exit_code = 1

    base = output_base_dir(cfg)
    report = _write_report(cfg, results, exit_code, base)
    print(f"Gate report written to {report}")
    if exit_code:
        print(
            "Gates FAILED — do not run rollouts until every required gate "
            "passes. SKIPPED gates (warnings) must be run on the "
            "evaluation machine."
        )
    sys.exit(exit_code)


if __name__ == "__main__":
    gates()
