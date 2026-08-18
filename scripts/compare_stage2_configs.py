"""Compare stage-2 configs across methods (freeze, lambda_phase, monitor)."""

from __future__ import annotations

import re
from pathlib import Path


def main() -> None:
    methods = ["phaseforge", "phase_pretrain_random_router", "plain_encoder_phase_bootstrap",
               "warmstart_moe", "scratch_moe"]
    for m in methods:
        for part in ("part1", "part2"):
            d = Path("outputs") / part / "outputs" / m / "stage2" / "seed42"
            if not d.is_dir():
                continue
            runs = sorted(d.iterdir())
            if not runs:
                continue
            txt = runs[0].joinpath("resolved_config.yaml").read_text()
            freeze = re.search(r"freeze_encoder: (\w+)", txt)
            lam = re.search(r"lambda_phase: ([\d.]+)", txt)
            nph = re.search(r"num_phases: (\d+)", txt)
            mon = re.search(r"monitor: (\S+)", txt)
            st1 = re.search(r"stage1_ckpt_path: (.+)", txt)
            topk = re.search(r"top_k: (\d+)", txt)
            noise = re.search(r"noise_std: ([\d.]+)", txt)
            epochs = re.search(r"epochs: (\d+)", txt)
            print(f"{m:>35} [{part}]: freeze={freeze.group(1) if freeze else '?'} "
                  f"lambda_phase={lam.group(1) if lam else '?'} num_phases={nph.group(1) if nph else '?'} "
                  f"top_k={topk.group(1) if topk else '?'} noise_std={noise.group(1) if noise else '?'} "
                  f"epochs={epochs.group(1) if epochs else '?'} mon={mon.group(1) if mon else '?'} "
                  f"stage1_ckpt={'yes' if st1 else 'no'}")
            break


if __name__ == "__main__":
    main()