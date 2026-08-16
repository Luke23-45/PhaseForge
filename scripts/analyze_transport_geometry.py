"""Dataset-wide geometry statistics for the Transport handover.

Reads every demo's INITIAL state from the frozen robomimic transport
dataset and reports the exact object/robot pose distributions so the
controller's meeting-point and reachability logic can be sized against
data instead of guesses.

Usage:  uv run python scripts/analyze_transport_geometry.py --all
"""

from __future__ import annotations

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

DATASET = r"data\raw\robomimic\transport\low_dim_v15.hdf5"
PAYLOAD_HEAD_OFFSET_Z = 0.117767
PANDA_REACH = 0.85
BASE0 = np.array([0.0, -0.25, 0.0])
BASE1 = np.array([0.0, +0.25, 0.0])
LIFT_Z = 1.075


def quat_to_rot(q_xyzw: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.array([q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]]))


def initial_of(demo) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obj = demo["obs"]["object"][0]
    eef0 = demo["obs"]["robot0_eef_pos"][0]
    eef1 = demo["obs"]["robot1_eef_pos"][0]
    g0 = demo["obs"]["robot0_gripper_qpos"][0]
    g1 = demo["obs"]["robot1_gripper_qpos"][0]
    return obj, eef0, eef1, g0, g1


def report():
    rows = []
    with h5py.File(DATASET, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for name in demos:
            obj, eef0, eef1, g0, g1 = initial_of(f["data"][name])
            payload = obj[0:3]
            quat = obj[3:7]
            R = quat_to_rot(quat)
            head_center = payload + R.apply([0.0, 0.0, PAYLOAD_HEAD_OFFSET_Z])
            # Match ScriptedTransportController._payload_meeting_point().
            meeting_x = float(np.clip(head_center[0], -0.35, 0.35))
            meeting_y = float(head_center[1])
            meeting_z = max(head_center[2] + 0.06, 0.90)
            meeting = np.array([meeting_x, meeting_y, meeting_z])
            d1 = np.linalg.norm(meeting - eef1)
            # head center vs meeting (wrist) offset:
            wrist_to_head = np.linalg.norm(meeting - head_center)
            # arm1 horizontal reach to meeting:
            reach1_xy = np.linalg.norm(meeting[:2] - BASE1[:2])
            # arm0 to head:
            reach0_head = np.linalg.norm(head_center[:2] - BASE0[:2])
            rows.append(
                dict(
                    demo=name,
                    payload=payload,
                    head_center=head_center,
                    eef1=eef1,
                    meeting=meeting,
                    d1=d1,
                    wrist_to_head=wrist_to_head,
                    reach1_xy=reach1_xy,
                    reach0_head_xy=reach0_head,
                    g1=g1,
                )
            )

    arr = {k: np.array([r[k] for r in rows]) for k in rows[0].keys()}

    def stats(name, col):
        v = arr[name]
        if isinstance(v, np.ndarray) and v.ndim > 1:
            for i, lbl in enumerate(col):
                c = v[:, i]
                print(
                    f"  {name}[{lbl}]: min={c.min():+.4f} p50={np.median(c):+.4f} "
                    f"max={c.max():+.4f}"
                )
        else:
            print(f"  {name}: min={v.min():+.4f} p50={np.median(v):+.4f} max={v.max():+.4f}")

    print(f"n demos = {len(rows)}")
    print("\npayload pos:")
    stats("payload", ["x", "y", "z"])
    print("\nhead_center (world):")
    stats("head_center", ["x", "y", "z"])
    print("\neef1 (arm1 wrist at reset):")
    stats("eef1", ["x", "y", "z"])
    print("\nmeeting point (controller convention):")
    stats("meeting", ["x", "y", "z"])
    print("\nscalars:")
    stats("d1", None)
    stats("wrist_to_head", None)
    stats("reach1_xy", None)
    stats("reach0_head_xy", None)

    print("\n--- head center vs meeting (wrist) relationship ---")
    head = arr["head_center"]
    meet = arr["meeting"]
    delta = head - meet
    for i, lbl in enumerate(["x", "y", "z"]):
        c = delta[:, i]
        print(f"  head-meeting [{lbl}]: min={c.min():+.4f} p50={np.median(c):+.4f} max={c.max():+.4f}")

    print("\n--- how far is the meeting point above the head top? ---")
    # head top z = head_center z + half head height (0.00825)
    top_delta = meet[:, 2] - (head[:, 2] + 0.00825)
    print(f"  meeting.z - head_top.z: min={top_delta.min():+.4f} p50={np.median(top_delta):+.4f} max={top_delta.max():+.4f}")

    print("\n--- how far above the PAYLOAD body is the meeting? ---")
    pz = arr["payload"][:, 2]
    print(f"  meeting.z - payload.z: min={(meet[:,2]-pz).min():+.4f} p50={np.median(meet[:,2]-pz):+.4f} max={(meet[:,2]-pz).max():+.4f}")

    print("\n--- X clamp distortion (head_center.x outside +/-0.35) ---")
    px = arr["head_center"][:, 0]
    mx = meet[:, 0]
    clamp_hit = np.abs(px) > 0.35
    print(f"  n head centers with |x|>0.35: {clamp_hit.sum()} / {len(px)}")
    if clamp_hit.any():
        print(f"  head_center.x max={px.max():+.4f} min={px.min():+.4f}")
        print(f"  meeting.x stays in [-0.35,0.35]; x distortion up to {np.abs(px - mx).max():.4f}")

    # wrist-to-head horizontal offset (the "did arm1 aim at the head" metric)
    print("\n--- wrist-to-head horizontal offset (x,y only) ---")
    hxy = head[:, :2] - meet[:, :2]
    print(f"  dx: min={hxy[:,0].min():+.4f} p50={np.median(hxy[:,0]):+.4f} max={hxy[:,0].max():+.4f}")
    print(f"  dy: min={hxy[:,1].min():+.4f} p50={np.median(hxy[:,1]):+.4f} max={hxy[:,1].max():+.4f}")
    hnorm = np.linalg.norm(hxy, axis=1)
    print(f"  |horiz|: min={hnorm.min():.4f} p50={np.median(hnorm):.4f} max={hnorm.max():.4f}")

    # arm1 reachability: can it reach the MEETING point (its commanded target)?
    print("\n--- arm1 horizontal reach to meeting vs PANDA_REACH ---")
    r1 = arr["reach1_xy"]
    print(f"  reach1_xy: min={r1.min():.4f} p50={np.median(r1):.4f} max={r1.max():.4f}")
    print(f"  n exceeding {PANDA_REACH:.2f}: {(r1 > PANDA_REACH).sum()}")

    print("\n--- arm0 horizontal reach to head center ---")
    r0 = arr["reach0_head_xy"]
    print(f"  reach0_head_xy: min={r0.min():.4f} p50={np.median(r0):.4f} max={r0.max():.4f}")
    print(f"  n exceeding {PANDA_REACH:.2f}: {(r0 > PANDA_REACH).sum()}")


if __name__ == "__main__":
    report()
