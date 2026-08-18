"""Cross-seed per-episode failure analysis for PhaseForge (Lift, shared reset bank)."""

from __future__ import annotations

import json
from pathlib import Path


def load_episodes(method: str, seed: str) -> list[dict]:
    for part in ("part1", "part2"):
        d = Path("outputs") / part / "outputs" / "eval" / method / f"seed{seed}"
        if d.is_dir():
            run = sorted(d.iterdir())[0]
            return [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines()]
    raise FileNotFoundError(f"{method} seed{seed}")


def main() -> None:
    p42 = load_episodes("phaseforge", "42")
    p43 = load_episodes("phaseforge", "43")
    p44 = load_episodes("phaseforge", "44")
    scr44 = load_episodes("scratch_moe", "44")
    boot44 = load_episodes("plain_encoder_phase_bootstrap", "44")

    assert len(p42) == len(p43) == len(p44) == 50

    s42 = {e["episode_index"]: e for e in p42}
    s43 = {e["episode_index"]: e for e in p43}
    s44 = {e["episode_index"]: e for e in p44}

    succ = lambda es: es["success"]

    # Episodes where s44 fails but s42 or s43 succeeds
    print("ep_idx  s42  s43  s44 | scratch44 boot44")
    lost = []
    for i in range(50):
        a, b, c = succ(s42[i]), succ(s43[i]), succ(s44[i])
        if c and not (a or b):
            print(f"  {i:>3}   {int(a)}    {int(b)}    {int(c)}   | (s44-only success)")
        if not c and (a or b):
            lost.append((i, a, b, c))
            s44_scr = succ(scr44[i]) if i < len(scr44) else None
            s44_boot = succ(boot44[i]) if i < len(boot44) else None
            print(f"  {i:>3}   {int(a)}    {int(b)}    {int(c)}   | {s44_scr} {s44_boot}")

    print()
    print(f"Episodes s44 failed but s42/s43 succeeded: {len(lost)}")
    print(f"  (of these, episode counts)")
    only_both = sum(1 for _, a, b, _ in lost if a and b)
    only_a = sum(1 for _, a, b, _ in lost if a and not b)
    only_b = sum(1 for _, a, b, _ in lost if not a and b)
    print(f"  succeeded in BOTH s42+s43: {only_both}")
    print(f"  succeeded only s42: {only_a}, only s43: {only_b}")
    print()

    # How stable is the "hard" set? episodes all methods fail.
    print("Episodes failed by ALL methods/seeds above (hard):")
    hard = []
    for i in range(50):
        all_fail = not succ(s42[i]) and not succ(s43[i]) and not succ(s44[i])
        all_fail &= (not succ(scr44[i])) and (not succ(boot44[i]))
        if all_fail:
            hard.append(i)
    print(" ", hard)


if __name__ == "__main__":
    main()