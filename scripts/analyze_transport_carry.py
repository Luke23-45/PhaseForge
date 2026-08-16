"""Precise arm0 reachability to target_bin using demo ground-truth.

The demos show arm0 carries the hammer across the gap directly to the
target_bin (158/200 demos) with no mid-air handover.  This script computes
the exact distance from arm0's base to the target_bin and the final
placement pose used in every demo, so we can size the transport waypoint
correctly.
"""

from __future__ import annotations

import h5py
import numpy as np

DATASET = r"data\raw\robomimic\transport\low_dim_v15.hdf5"
BASE0 = np.array([0.0, -0.25, 0.0])
BASE1 = np.array([0.0, +0.25, 0.0])
PANDA_REACH = 0.85


def report() -> None:
    rows = []
    with h5py.File(DATASET, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for name in demos:
            demo = f["data"][name]
            obj = demo["obs"]["object"][:]
            eef0 = demo["obs"]["robot0_eef_pos"][:]
            target_bin = obj[0, 21:24]
            payload = obj[:, 0:3]
            last = payload[-1]

            # arm0 distance from base to target_bin
            d0_bin = np.linalg.norm(target_bin[:2] - BASE0[:2])
            # arm0 eef horizontal reach from base over the whole traj
            reach0 = np.linalg.norm(eef0[:, :2] - BASE0[:2], axis=1)
            # final eef0
            eef0_last = eef0[-1]

            rows.append(
                dict(
                    demo=name,
                    d0_bin=d0_bin,
                    reach0_max=float(reach0.max()),
                    eef0_last=eef0_last,
                    payload_last=last,
                    target_bin=target_bin,
                )
            )

    d0_bin = np.array([r["d0_bin"] for r in rows])
    rmax = np.array([r["reach0_max"] for r in rows])
    print(f"n = {len(rows)}")
    print(f"arm0 base->target_bin horiz: min={d0_bin.min():.3f} p50={np.median(d0_bin):.3f} max={d0_bin.max():.3f}")
    print(f"  all within 0.85m? {(d0_bin <= PANDA_REACH).all()}")
    print(f"arm0 eef max horiz reach from base: min={rmax.min():.3f} p50={np.median(rmax):.3f} max={rmax.max():.3f}")
    print(f"  all within 0.85m? {(rmax <= PANDA_REACH).all()}")

    # Final eef0 & payload distribution
    e0l = np.array([r["eef0_last"] for r in rows])
    pl = np.array([r["payload_last"] for r in rows])
    print("\nfinal eef0:")
    for i, lbl in enumerate(["x", "y", "z"]):
        c = e0l[:, i]
        print(f"  [{lbl}]: min={c.min():+.3f} p50={np.median(c):+.3f} max={c.max():+.3f}")
    print("final payload:")
    for i, lbl in enumerate(["x", "y", "z"]):
        c = pl[:, i]
        print(f"  [{lbl}]: min={c.min():+.3f} p50={np.median(c):+.3f} max={c.max():+.3f}")

    # distance from final eef0 to final payload (the carry offset)
    off = e0l - pl
    print("\neef0 - payload offset at final step:")
    for i, lbl in enumerate(["x", "y", "z"]):
        c = off[:, i]
        print(f"  [{lbl}]: min={c.min():+.3f} p50={np.median(c):+.3f} max={c.max():+.3f}")
    print(f"  |offset|: min={np.linalg.norm(off,axis=1).min():.3f} "
          f"p50={np.median(np.linalg.norm(off,axis=1)):.3f} "
          f"max={np.linalg.norm(off,axis=1).max():.3f}")

    # is the final payload inside the target_bin footprint?  bin ~ 0.2x0.2m
    print("\nfinal payload vs target_bin (footprint ~0.2m):")
    for i, lbl in enumerate(["x", "y"]):
        c = pl[:, i] - np.array([r["target_bin"][i] for r in rows])
        print(f"  delta[{lbl}]: min={c.min():+.3f} p50={np.median(c):+.3f} max={c.max():+.3f}")

    # how high does arm0 lift the payload during transport?  (mid traj z)
    print("\npayload z trajectory (carry height):")
    zs = []
    with h5py.File(DATASET, "r") as f:
        for name in demos:
            obj = f["data"][name]["obs"]["object"][:]
            pz = obj[:, 2]
            zs.append(pz)
    allz = np.concatenate(zs)
    print(f"  payload z all-steps: min={allz.min():.3f} p50={np.median(allz):.3f} max={allz.max():.3f}")
    # when z>0.95 (in flight)
    print(f"  fraction of steps with z>0.95: {(allz>0.95).mean():.3f}")


if __name__ == "__main__":
    report()