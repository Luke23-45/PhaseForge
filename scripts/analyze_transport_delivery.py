"""Trace the payload delivery mechanism in a demo end-to-end.

Determines which arm physically carries the hammer to the target_bin by
tracking, at every step, which eef is nearest to the payload body, and
summarises the final 150 steps where the delivery happens.
"""

from __future__ import annotations

import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

DATASET = r"data\raw\robomimic\transport\low_dim_v15.hdf5"
PAYLOAD_HEAD_OFFSET_Z = 0.115


def quat_to_rot(q_xyzw):
    return Rotation.from_quat(np.array([q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]]))


def trace(idx: int) -> None:
    with h5py.File(DATASET, "r") as f:
        demo = f["data"][f"demo_{idx}"]
        obj = demo["obs"]["object"][:]
        eef0 = demo["obs"]["robot0_eef_pos"][:]
        eef1 = demo["obs"]["robot1_eef_pos"][:]
        g0 = demo["obs"]["robot0_gripper_qpos"][:]
        g1 = demo["obs"]["robot1_gripper_qpos"][:]
        n = obj.shape[0]

    payload = obj[:, 0:3]
    head = np.stack(
        [p + quat_to_rot(q).apply([0, 0, PAYLOAD_HEAD_OFFSET_Z]) for p, q in zip(payload, obj[:, 3:7])]
    )
    target_bin = obj[0, 21:24]

    d0 = np.linalg.norm(eef0 - payload, axis=1)
    d1 = np.linalg.norm(eef1 - payload, axis=1)
    nearest = np.where(d0 < d1, 0, 1)

    g0_open = np.abs(g0[:, 0]) + np.abs(g0[:, 1])
    g1_open = np.abs(g1[:, 0]) + np.abs(g1[:, 1])

    # who carries across the gap (payload y>0)?
    carry = np.where(payload[:, 1] > 0.0, nearest, -1)

    print(f"=== demo_{idx} (n={n}) payload delivery ===")
    print(f"target_bin = {target_bin}")
    print(f"\npayload y crosses 0 at step {np.argmax(payload[:,1] > 0.0)} (first +y)")
    # arm that is nearest while payload y>0
    pos = carry[carry != -1]
    if pos.size:
        print(f"arm nearest to payload while y>0: arm0={ (pos==0).sum() } steps, "
              f"arm1={ (pos==1).sum() } steps")

    # Sample the last 60 steps: who's near, where's payload
    print("\nlast 60 steps (delivery):")
    print("  t    payload(x,y,z)     eef0(x,y)  eef1(x,y)  nearest g0 g1")
    for j in range(max(0, n - 60), n):
        g0c = "C" if g0_open[j] < 0.01 else "O"
        g1c = "C" if g1_open[j] < 0.01 else "O"
        print(f"  {j:4d} ({payload[j,0]:+.3f},{payload[j,1]:+.3f},{payload[j,2]:+.3f}) "
              f"({eef0[j,0]:+.3f},{eef0[j,1]:+.3f}) ({eef1[j,0]:+.3f},{eef1[j,1]:+.3f}) "
              f"{nearest[j]} {g0c} {g1c}")

    # what was the max y the payload reached, and who had it?
    print(f"\npayload y max = {payload[:,1].max():+.3f} at t={np.argmax(payload[:,1])}")
    print(f"payload z max = {payload[:,2].max():+.3f} at t={np.argmax(payload[:,2])}")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    trace(idx)