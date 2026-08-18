"""Wave C2: failure analysis from rollout artifacts.

Reads every ``rollout_summary.json`` and ``episodes.jsonl`` under the
given eval directories and aggregates:
- per-method success rate, mean steps-to-failure, failure-category mix
- per-method phase distribution of *successful* episodes' final phase
  (state.jsonl does not record per-step phases; we can only report the
  episode-final step count, which is a weak proxy for failure-timestep)

The honest conclusion of this script: per-step phase attribution for
failed rollouts is not available without instrumented re-runs. The
artifact-only analysis below quantifies how informative that gap is.

Outputs:
    outputs/surgical/_findings/failure_phase.json
    docs/dev/findings/failure_phase.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FINDINGS_DIR = Path("outputs/surgical/_findings")
REPORT_PATH = Path("docs/dev/findings/failure_phase.md")


def _gather(eval_dir: Path) -> list[dict]:
    out = []
    for summary_path in eval_dir.rglob("rollout_summary.json"):
        run_dir = summary_path.parent
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        ep_path = run_dir / "episodes.jsonl"
        ep_steps = []
        ep_successes = []
        ep_categories = []
        if ep_path.is_file():
            for line in ep_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ep_steps.append(int(row.get("steps", 0)))
                ep_successes.append(bool(row.get("success", False)))
                ep_categories.append(row.get("failure_category"))
        out.append({
            "model": data.get("model"),
            "tag": data.get("tag"),
            "seed": data.get("training_seed"),
            "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
            "horizon": data.get("horizon"),
            "episodes": data.get("episodes"),
            "summary_sr": (data.get("metrics", {}) or {}).get("eval/rollout/success_rate"),
            "failure_categories": data.get("failure_categories") or {},
            "ep_steps": ep_steps,
            "ep_successes": ep_successes,
            "ep_categories": ep_categories,
        })
    return out


def _aggregate(records: list[dict]) -> dict:
    by_model_tag: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (r["model"], r["tag"])
        slot = by_model_tag.setdefault(key, {
            "model": r["model"], "tag": r["tag"], "n_runs": 0,
            "episodes": 0, "successes": 0,
            "category_counter": Counter(),
            "steps_success": [],
            "steps_failure": [],
        })
        slot["n_runs"] += 1
        slot["episodes"] += r["episodes"] or 0
        slot["successes"] += sum(1 for s in r["ep_successes"] if s)
        for cat, count in (r["failure_categories"] or {}).items():
            slot["category_counter"][cat] += count
        for steps, success in zip(r["ep_steps"], r["ep_successes"]):
            (slot["steps_success"] if success else slot["steps_failure"]).append(steps)
    summary = []
    for slot in by_model_tag.values():
        n_eps = max(slot["episodes"], 1)
        summary.append({
            "model": slot["model"],
            "tag": slot["tag"],
            "n_runs": slot["n_runs"],
            "episodes": slot["episodes"],
            "success_rate": slot["successes"] / n_eps,
            "failure_category_breakdown": dict(slot["category_counter"]),
            "mean_steps_success": (sum(slot["steps_success"]) / len(slot["steps_success"])) if slot["steps_success"] else None,
            "mean_steps_failure": (sum(slot["steps_failure"]) / len(slot["steps_failure"])) if slot["steps_failure"] else None,
        })
    summary.sort(key=lambda x: (x["model"], x["tag"] or ""))
    return {"by_method": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dirs", nargs="+", required=True,
                        help="Paths containing eval run dirs (recursively scanned for rollout_summary.json).")
    parser.add_argument("--label", required=True, help="Short label written into the report header.")
    args = parser.parse_args(argv)

    all_records = []
    for raw in args.eval_dirs:
        d = (PROJECT_ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        if not d.is_dir():
            print(f"[fail-phase] skip {d}: not a directory")
            continue
        recs = _gather(d)
        print(f"[fail-phase] {d}: {len(recs)} eval runs")
        all_records.extend(recs)
    agg = _aggregate(all_records)
    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": args.label,
        "eval_dirs": list(args.eval_dirs),
        "n_eval_runs": len(all_records),
        **agg,
        "honesty": "Per-step phase annotations are NOT stored in the rollout artifacts; "
                   "this script only reports success/failure mix and step counts. Per-phase "
                   "failure attribution requires an instrumented rollout re-run (state + phase "
                   "captured per timestep), which is out of scope for the current artifacts.",
    }
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    findings_path = PROJECT_ROOT / FINDINGS_DIR / f"failure_phase_{args.label}.json"
    findings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _render_report(payload)
    print(f"[fail-phase] done -> {findings_path}")
    return 0


def _render_report(payload: dict) -> None:
    lines = [
        f"# Wave C2 — Failure analysis ({payload['label']})",
        "",
        f"Created {payload['created']} on branch `surgical-cpu-analysis`.",
        "",
        payload["honesty"],
        "",
        "| model | tag | n_runs | episodes | success_rate | mean steps success | mean steps failure |",
        "|---|---|---|---|---|---|---|",
    ]
    for slot in payload["by_method"]:
        ms = slot["mean_steps_success"]
        mf = slot["mean_steps_failure"]
        ms_str = f"{ms:.1f}" if ms is not None else "-"
        mf_str = f"{mf:.1f}" if mf is not None else "-"
        lines.append(
            f"| {slot['model']} | {slot['tag']} | {slot['n_runs']} | {slot['episodes']} | "
            f"{slot['success_rate']:.3f} | {ms_str} | {mf_str} |"
        )
    lines += ["", "## Failure category breakdown (sum across runs)", ""]
    for slot in payload["by_method"]:
        if not slot["failure_category_breakdown"]:
            continue
        cats = ", ".join(f"{k}={v}" for k, v in sorted(slot["failure_category_breakdown"].items(), key=lambda kv: -kv[1]))
        lines.append(f"- **{slot['model']}** `{slot['tag']}`: {cats}")
    lines += ["", "## Interpretation", "", "- The 'mean steps failure' column reveals whether failures cluster at the horizon (= task_timeout) or earlier.", ""]
    report = PROJECT_ROOT / REPORT_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
