"""Analyze ground-truth demos: how do arm0/arm1 actually move the hammer?

The robomimic dataset contains human/expert demonstrations of Transport.
We extract the eef trajectories and payload motion to see the actual
solution pattern (who carries the hammer, where the head ends up, the
grasp poses used) so the scripted controller can mirror the ground truth
instead of guessing.

Usage:  uv run python scripts/analyze_transport_demos.py [demo_idx] [--summary]
"""

from __future__ import annotations

import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

DATASET = r"data\raw\robomimic\transport\low_dim_v15.hdf5"
PAYLOAD_HEAD_OFFSET_Z = 0.115
LIFT_Z = 1.075


def quat_to_rot(q_xyzw: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.array([q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]]))


def trace_demo(demo_idx: int, verbose: bool = True) -> dict:
    with h5py.File(DATASET, "r") as f:
        demo = f["data"][f"demo_{demo_idx}"]
        obj = demo["obs"]["object"][:]
        eef0 = demo["obs"]["robot0_eef_pos"][:]
        eef1 = demo["obs"]["robot1_eef_pos"][:]
        g0 = demo["obs"]["robot0_gripper_qpos"][:]
        g1 = demo["obs"]["robot1_gripper_qpos"][:]
        n = obj.shape[0]

    payload = obj[:, 0:3]
    pquat = obj[:, 3:7]
    trash = obj[:, 7:10]
    target_bin = obj[0, 21:24]
    trash_bin = obj[0, 24:27]

    # head center in world at every step
    head = np.stack(
        [p + quat_to_rot(q).apply([0, 0, PAYLOAD_HEAD_OFFSET_Z]) for p, q in zip(payload, pquat)]
    )

    # gripper openness (0 = closed)
    g0_open = np.abs(g0[:, 0]) + np.abs(g0[:, 1])
    g1_open = np.abs(g1[:, 0]) + np.abs(g1[:, 1])

    # when does each arm close its gripper? (openness drops)
    g0_closed = g0_open < 0.01
    g1_closed = g1_open < 0.01

    # find contiguous closed windows for each arm
    def windows(mask):
        out = []
        start = None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                out.append((start, i))
                start = None
        if start is not None:
            out.append((start, len(mask)))
        return out

    w0 = windows(g0_closed)
    w1 = windows(g1_closed)

    # arm0 payload contact windows: when is eef0 near payload?
    d0p = np.linalg.norm(eef0 - payload, axis=1)
    near0 = d0p < 0.15
    d1h = np.linalg.norm(eef1 - head, axis=1)

    if verbose:
        print(f"=== demo_{demo_idx} (n={n} steps) ===")
        print(f"target_bin = {target_bin}  trash_bin = {trash_bin}")
        print(f"\narm0 closed-gripper windows: {w0}")
        print(f"arm1 closed-gripper windows: {w1}")
        print(f"\narm0 eef within 0.15m of payload body: "
              f"{(near0).sum()} steps (windows {windows(near0)[:8]})")

        # When arm1's eef is near the head:
        near1 = d1h < 0.15
        print(f"arm1 eef within 0.15m of HEAD: {(near1).sum()} steps (windows {windows(near1)[:8]})")

        # z trajectory of the payload (lift height)
        print(f"\npayload z: start={payload[0,2]:.3f} min={payload[:,2].min():.3f} "
              f"max={payload[:,2].max():.3f} end={payload[-1,2]:.3f}")
        print(f"payload y: start={payload[0,1]:+.3f} end={payload[-1,1]:+.3f} (target_bin y={target_bin[1]:+.3f})")
        print(f"payload x: start={payload[0,0]:+.3f} end={payload[-1,0]:+.3f} (target_bin x={target_bin[0]:+.3f})")

        # what z does arm1 use when grasping the head?
        z_near1 = eef1[near1, 2] if near1.any() else np.array([])
        if z_near1.size:
            print(f"\narm1 eef z when near head: min={z_near1.min():.3f} "
                  f"p50={np.median(z_near1):.3f} max={z_near1.max():.3f}")
            print(f"head z when arm1 near: min={head[near1,2].min():.3f} "
                  f"p50={np.median(head[near1,2]):.3f} max={head[near1,2].max():.3f}")

        # eef0 z when holding payload
        z_near0 = eef0[near0, 2] if near0.any() else np.array([])
        if z_near0.size:
            print(f"\narm0 eef z when near payload: min={z_near0.min():.3f} "
                  f"p50={np.median(z_near0):.3f} max={z_near0.max():.3f}")

        # Does arm1 grasp the HEAD or the HANDLE?  Check eef1 xy vs head xy and payload xy
        print("\narm1 wrist xy vs head xy and payload xy (when within 0.15m of head):")
        if near1.any():
            idx = np.where(near1)[0]
            for j in idx[:: max(1, len(idx) // 10)][:10]:
                hx, hy = head[j, 0], head[j, 1]
                px, py = payload[j, 0], payload[j, 1]
                ex, ey = eef1[j, 0], eef1[j, 1]
                print(f"  t={j}: eef1=({ex:+.3f},{ey:+.3f}) head=({hx:+.3f},{hy:+.3f}) "
                      f"payload=({px:+.3f},{py:+.3f}) z_eef={eef1[j,2]:.3f} z_head={head[j,2]:.3f}")

    return dict(
        w0=w0, w1=w1, d0p=d0p, d1h=d1h, payload=payload, head=head,
        eef0=eef0, eef1=eef1, g0_closed=g0_closed, g1_closed=g1_closed,
    )


def summary() -> None:
    stats = {"n": 0, "arm0_holds_payload_past_y0": 0, "arm1_holds_payload": 0,
             "arm0_carries_across": 0, "payload_ends_in_bin": 0}
    reach0_max = []
    reach1_head = []
    with h5py.File(DATASET, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for name in demos:
            idx = int(name.split("_")[1])
            t = trace_demo(idx, verbose=False)
            stats["n"] += 1
            payload = t["payload"]
            head = t["head"]
            eef0, eef1 = t["eef0"], t["eef1"]
            # does payload end near target_bin x/y?
            last = payload[-1]
            target_bin_x = last[0]
            # crude: payload final x within 0.1 of initial (symmetric)?  Use y sign.
            if last[1] > 0:  # ended on the +y (right) side
                stats["payload_ends_in_bin"] += 1
            # arm0 max horizontal reach used
            reach0_max.append(float(np.max(np.linalg.norm(eef0[:, :2], axis=1))))
            # arm1's wrist-to-head distance when closest
            d1h = t["d1h"]
            reach1_head.append(float(np.min(d1h)))
            # does arm0 carry the payload across the gap?  (payload y crosses 0 while arm0 near)
            d0p = t["d0p"]
            near0 = d0p < 0.15
            py = payload[:, 1]
            if near0.any() and (py[near0].max() > 0.0):
                stats["arm0_carries_across"] += 1

    print(f"n demos = {stats['n']}")
    print(f"payload ends on +y (right/target) side: {stats['payload_ends_in_bin']}")
    print(f"arm0 carries payload across gap (payload y>0 while arm0 near): {stats['arm0_carries_across']}")
    print(f"arm0 max horizontal reach used: min={np.min(reach0_max):.3f} "
          f"p50={np.median(reach0_max):.3f} max={np.max(reach0_max):.3f}")
    print(f"arm1 min wrist-to-head distance: min={np.min(reach1_head):.3f} "
          f"p50={np.median(reach1_head):.3f} max={np.max(reach1_head):.3f}")


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary()
    else:
        idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        trace_demo(idx)