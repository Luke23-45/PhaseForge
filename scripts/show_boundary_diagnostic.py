"""Print the phase-boundary-error breakdown for the λ-decay checkpoints."""

from __future__ import annotations

import glob
import json

ROOT = "outputs_local_train/phaseforge/stage1"


def main() -> None:
    for s in ("42", "43", "44"):
        path = glob.glob(f"{ROOT}/seed{s}/*lambdav1*/phase_boundary_diagnostic.json")[0]
        d = json.load(open(path))
        keys = sorted(k for k in d if k.startswith("eval/phase_err_dist_"))
        label = lambda k: k.replace("eval/phase_err_dist_", "")  # noqa: E731
        rates = [f"{label(k)}={d[k]:.2f}" for k in keys]
        print(f"seed{s}: " + "  ".join(rates))
        print(
            f"        n_boundaries={d['eval/phase_err_n_boundaries']:.0f} "
            f"n_samples={d['eval/phase_err_n_samples']:.0f} "
            f"any_boundary={d['eval/phase_err_any_boundary']:.2f}"
        )


if __name__ == "__main__":
    main()