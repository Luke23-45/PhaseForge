"""Validate the corrected head-center meeting point against all 200 demos.

The controller currently aims arm1's wrist at ``payload + [0,0,0.115]``
(assuming the head is directly above the body in world-Z), but the hammer
lies flat, so the real head is offset along the hammer's local axis
(horizontal).  This script computes the corrected meeting point from the
payload quat for every demo's handover window and checks arm1's reach.

It also verifies the ground-truth: arm1's wrist is ~5cm above the head
center when it grasps, with the fingers descending through the head.
"""

from __future__ import annotations

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

DATASET = r"data\raw\robomimic\transport\low_dim_v15.hdf5"
PAYLOAD_HEAD_OFFSET_Z = 0.115
BASE1 = np.array([0.0, +0.25, 0.0])
PANDA_REACH = 0.85
# wrist above head center when grasping (from demo_0: eef1 z 1.068 vs head z 1.016)
WRIST_ABOVE_HEAD = 0.05
# finger length below wrist
FINGER_LEN = 0.10


def quat_to_rot(q_xyzw):
    return Rotation.from_quat(np.array([q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]]))


def head_center_world(payload, quat):
    return payload + quat_to_rot(quat).apply([0, 0, PAYLOAD_HEAD_OFFSET_Z])


def report() -> None:
    rows = []
    with h5py.File(DATASET, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for name in demos:
            demo = f["data"][name]
            obj = demo["obs"]["object"][:]
            eef1 = demo["obs"]["robot1_eef_pos"][:]
            n = obj.shape[0]
            payload = obj[:, 0:3]
            pquat = obj[:, 3:7]
            head = np.stack(
                [head_center_world(p, q) for p, q in zip(payload, pquat)]
            )
            # corrected meeting point = head center + wrist-above-head in z
            corrected = head + np.array([0.0, 0.0, WRIST_ABOVE_HEAD])

            # find the handover: the closest approach of arm1 to the head
            d1h = np.linalg.norm(eef1 - head, axis=1)
            if d1h.size:
                j = int(np.argmin(d1h))
                reach_at = np.linalg.norm(corrected[j, :2] - BASE1[:2])
                wrist_z = eef1[j, 2]
                head_z = head[j, 2]
                wrist_above = wrist_z - head_z
            else:
                j = n - 1
                wrist_above = None

            rows.append(
                dict(
                    demo=name,
                    handover_steps=0,
                    reach_at=reach_at,
                    wrist_above=wrist_above,
                    head_z=head[j, 2],
                    payload_z=payload[j, 2],
                )
            )

    hs = np.array([r["handover_steps"] for r in rows])
    print(f"n = {len(rows)}")

    contact = rows  # all demos use closest-approach now
    reach = np.array([r["reach_at"] for r in contact])
    wa = np.array([r["wrist_above"] for r in contact])
    hz = np.array([r["head_z"] for r in contact])
    if reach.size:
        print(f"\narm1 reach to corrected meeting (horizontal, from base y=+0.25):")
        print(f"  min={reach.min():.3f} p50={np.median(reach):.3f} max={reach.max():.3f}")
        print(f"  all within {PANDA_REACH}? {(reach <= PANDA_REACH).all()}")
    if wa.size:
        print(f"\nwrist z - head z at handover start (should be ~+0.05 for top-down grasp):")
        print(f"  min={wa.min():+.3f} p50={np.median(wa):+.3f} max={wa.max():+.3f}")

    print(f"\nhead z at handover start: min={hz.min():.3f} p50={np.median(hz):.3f} max={hz.max():.3f}")

    # finger envelope check: wrist z - finger_len vs head top
    print(f"\nfinger tip reaches (wrist_z - {FINGER_LEN}):")
    if wa.size:
        # wrist above = wa; wrist z = head z + wa; tip z = head z + wa - finger_len
        tip = hz + wa - FINGER_LEN
        print(f"  tip z - head center z: min={(tip-hz).min():+.3f} p50={np.median(tip-hz):+.3f} max={(tip-hz).max():+.3f}")


if __name__ == "__main__":
    report()