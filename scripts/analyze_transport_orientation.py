"""Hammer orientation during the ground-truth handover.

The demo shows arm0 carries the hammer high, arm1 grabs the HEAD from
above.  This computes, for every step, whether the hammer is vertical
(head above body, head below body) or flat, and where the head actually
is -- so the meeting point can be sized from data.
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
        n = obj.shape[0]
    payload = obj[:, 0:3]
    pquat = obj[:, 3:7]

    head_world = np.stack(
        [p + quat_to_rot(q).apply([0, 0, PAYLOAD_HEAD_OFFSET_Z]) for p, q in zip(payload, pquat)]
    )
    # vertical offset of head above body (world z)
    dz = head_world[:, 2] - payload[:, 2]
    # horizontal offset
    dxy = np.linalg.norm(head_world[:, :2] - payload[:, :2], axis=1)

    print(f"=== demo_{idx}: hammer orientation over time ===")
    print("  t    payload.z  head.z   head-payload.dz  head-payload.dxy  orientation")
    for j in range(0, n, 8):
        if payload[j, 1] > -0.3:  # once it starts crossing, focus
            continue
        o = "head-above" if dz[j] > 0.08 else ("head-below" if dz[j] < -0.08 else "flat")
        print(f"  {j:4d}  {payload[j,2]:+.3f}  {head_world[j,2]:+.3f}  {dz[j]:+.3f}  {dxy[j]:+.3f}  {o}")

    print("\n  after crossing to +y (carry/handover phase):")
    for j in range(0, n, 8):
        if payload[j, 1] <= -0.3:
            continue
        o = "head-above" if dz[j] > 0.08 else ("head-below" if dz[j] < -0.08 else "flat")
        print(f"  {j:4d}  {payload[j,2]:+.3f}  {head_world[j,2]:+.3f}  {dz[j]:+.3f}  {dxy[j]:+.3f}  {o}")

    print(f"\n  where is the head during the peak lift (z>1.0)?")
    high = payload[:, 2] > 1.0
    if high.any():
        dz_high = dz[high]
        print(f"  head-payload.dz at z>1.0: min={dz_high.min():+.3f} p50={np.median(dz_high):+.3f} "
              f"max={dz_high.max():+.3f}")
        print(f"  fraction head-above (dz>0.08) at z>1.0: {(dz_high>0.08).mean():.3f}")
        print(f"  fraction head-below (dz<-0.08) at z>1.0: {(dz_high<-0.08).mean():.3f}")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    trace(idx)