"""Validate the payload-body handover target against all Transport demos."""

from __future__ import annotations

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

DATASET = r"data\raw\robomimic\transport\low_dim_v15.hdf5"
BASE1 = np.array([0.0, +0.25, 0.0])
PANDA_REACH = 0.85
HANDOVER_Z_MIN = 0.90


def payload_close_index(actions: np.ndarray, payload: np.ndarray) -> int:
    """Find arm1's payload close transition after the payload is lifted."""
    transitions = np.where((actions[1:, 13] == 1) & (actions[:-1, 13] != 1))[0] + 1
    lifted = [
        int(t)
        for t in transitions
        if t > 0 and np.max(payload[max(0, t - 10) : t, 2]) > payload[0, 2] + 0.03
    ]
    if lifted:
        return lifted[0]
    if len(transitions) >= 2:
        return int(transitions[1])
    raise ValueError("demo has no identifiable arm1 payload close transition")


def report() -> None:
    rows: list[dict[str, np.ndarray | float | int | str]] = []
    with h5py.File(DATASET, "r") as h5:
        demos = sorted(h5["data"].keys(), key=lambda name: int(name.split("_")[1]))
        for name in demos:
            demo = h5["data"][name]
            actions = np.asarray(demo["actions"][:], dtype=np.float64)
            objects = np.asarray(demo["obs"]["object"][:], dtype=np.float64)
            eef1 = np.asarray(demo["obs"]["robot1_eef_pos"][:], dtype=np.float64)
            payload = objects[:, :3]
            index = payload_close_index(actions, payload)
            sample = min(index + 5, len(payload) - 1)
            target = payload[sample].copy()
            target[2] = max(target[2], HANDOVER_Z_MIN)
            offset_world = eef1[sample] - payload[sample]
            offset_payload = Rotation.from_quat(objects[sample, 3:7]).inv().apply(
                offset_world
            )
            rows.append(
                {
                    "demo": name,
                    "close_index": index,
                    "reach": float(np.linalg.norm(target[:2] - BASE1[:2])),
                    "offset_world": offset_world,
                    "offset_payload": offset_payload,
                }
            )

    reaches = np.asarray([row["reach"] for row in rows], dtype=np.float64)
    offsets = np.asarray([row["offset_payload"] for row in rows], dtype=np.float64)
    lateral = np.linalg.norm(offsets[:, :2], axis=1)
    print(f"n demos = {len(rows)}")
    print(
        "arm1 reach to payload-body target: "
        f"min={reaches.min():.3f} p50={np.median(reaches):.3f} "
        f"max={reaches.max():.3f}; all <= {PANDA_REACH}: "
        f"{bool(np.all(reaches <= PANDA_REACH))}"
    )
    for axis, values in zip(("x", "y", "z"), offsets.T):
        print(
            f"payload-frame EEF offset {axis}: "
            f"p05={np.quantile(values, .05):+.4f} "
            f"p50={np.median(values):+.4f} "
            f"p95={np.quantile(values, .95):+.4f}"
        )
    print(f"lateral offset <= 0.04 m: {int(np.sum(lateral <= 0.04))}/{len(rows)}")
    print(
        "handle-axis offset <= 0.10 m: "
        f"{int(np.sum(np.abs(offsets[:, 2]) <= 0.10))}/{len(rows)}"
    )


if __name__ == "__main__":
    report()
