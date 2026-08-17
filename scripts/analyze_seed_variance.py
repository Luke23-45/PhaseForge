"""Analyze per-seed success variance for phaseforge vs baselines.

Uses paired episodes (same reset bank / reset_seed across all evals) to
compare per-episode outcomes across training seeds and methods.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

OUTPUTS = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")


def load_episodes(path: Path):
    episodes = {}
    steps = {}
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            episodes[e["episode_index"]] = bool(e["success"])
            steps[e["episode_index"]] = e["steps"]
    return episodes, steps


def find_evals():
    evals = []
    for ep_file in OUTPUTS.rglob("episodes.jsonl"):
        run_dir = ep_file.parent
        method = run_dir.parent.parent.name
        seed = int(run_dir.parent.name.replace("seed", ""))
        evals.append((method, seed, run_dir))
    return evals


def main():
    evals = find_evals()
    by_method = defaultdict(dict)
    for method, seed, run_dir in sorted(evals):
        by_method[method][seed] = (run_dir, *load_episodes(run_dir / "episodes.jsonl"))

    print("=== per-method per-seed success rates ===")
    methods = sorted(by_method)
    for m in methods:
        seeds = sorted(by_method[m])
        rates = {s: sum(by_method[m][s][1].values()) / 50 for s in seeds}
        rates_str = "  ".join(f"seed{s}: {r:.2f}" for s, r in rates.items())
        vals = list(rates.values())
        spread = max(vals) - min(vals)
        print(f"{m:32s} {rates_str}  spread={spread:.2f}")

    print()
    print("=== phaseforge: per-episode agreement across seeds ===")
    pf = by_method.get("phaseforge", {})
    seeds = sorted(pf)
    if len(seeds) >= 2:
        n = 50
        eps = {s: [pf[s][1][i] for i in range(n)] for s in seeds}
        agree = sum(1 for i in range(n) if len({eps[s][i] for s in seeds}) == 1)
        print(f"episodes where all seeds agree: {agree}/50")
        for i in range(n):
            outcomes = {s: ("S" if eps[s][i] else "F") for s in seeds}
            if len(set(outcomes.values())) > 1:
                steps_str = "  ".join(f"{s}:{pf[s][2][i]}" for s in seeds)
                print(f"  ep{i:02d} " + "".join(outcomes.values()) + f"  steps: {steps_str}")

    print()
    print("=== cross-method agreement at episode level (per seed) ===")
    for seed in (42, 43, 44):
        print(f"-- seed {seed} --")
        method_eps = {}
        for m in methods:
            if seed in by_method[m]:
                method_eps[m] = [by_method[m][seed][1][i] for i in range(50)]
        agree = sum(
            1 for i in range(50) if len({e[i] for e in method_eps.values()}) == 1
        )
        print(f"  episodes where all methods agree: {agree}/50")
        disagree = [
            i
            for i in range(50)
            if len({e[i] for e in method_eps.values()}) > 1
        ]
        for i in disagree:
            outs = "  ".join(f"{m}:{'S' if method_eps[m][i] else 'F'}" for m in method_eps)
            print(f"  ep{i:02d}  {outs}")


if __name__ == "__main__":
    main()