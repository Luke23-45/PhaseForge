"""Static geometry verification for the Transport rollout failures.

Reconstructs the Panda-gripper fingerpad world positions from the exact
MuJoCo model (embedded in data/raw/robomimic/transport/low_dim_v15.hdf5)
and the log's per-step EEF pose/quat, then checks:

  1. Chain regression: computed open-pad distances vs the log's ground-truth
     ``min_dist`` (real ``geom_xpos`` from MuJoCo) at LID_GRASP entry.
  2. Case 00 (trash grasp miss) vs case 01 (success): where do the pads end
     up after closing (q=0), does the sweep bracket the 4cm cube?
  3. Case 01 handover: the frozen meeting target vs payload geometry at
     TABLE_DESCEND entry, and the pad contact point on the payload.

MuJoCo XML quat attributes are (w, x, y, z); the state/log quats are
(x, y, z, w).
"""

import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\Hellx\Documents\Programming\python\Project\Neryva\PhaseForge")
LOG = ROOT / "debug_runs_logs.md"

# --- exact gripper chain from the embedded MJCF (w,x,y,z quats) -------------
# right_hand body: pos (0,0,0.1065) quat Rz(45) relative to link7
# right_gripper body: pos 0 quat Rz(-90) relative to right_hand
# eef body: pos (0,0,0.097) in right_gripper  (grip site at origin)
# leftfinger body: pos (0,0,0.0524) quat Rz(+90) in right_gripper
#   joint1 axis (0,1,0) range [0, 0.04]
#   tip1 body: pos (0, 0.0085, 0.056); pad1 geom pos (0, -0.005, -0.015)
# rightfinger body: pos (0,0,0.0524) quat Rz(+90) in right_gripper
#   joint2 axis (0,1,0) range [-0.04, 0]
#   tip2 body: pos (0, -0.0085, 0.056); pad2 geom pos (0, +0.005, -0.015)
#
# pad centers in right_gripper frame:
#   pad1 = (-0.0035 - q1, 0, 0.0934)   q1 in [0, 0.04]
#   pad2 = ( 0.0035 - q2, 0, 0.0934)   q2 in [-0.04, 0]
# eef site in right_gripper frame: (0, 0, 0.097)

PAD1_OFFSET_E = np.array([-0.0035, 0.0, -0.0036])
PAD2_OFFSET_E = np.array([0.0035, 0.0, -0.0036])
PAD_SLIDE = np.array([-1.0, 0.0, 0.0])


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(q))
    if norm == 0:
        return np.eye(3)
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pad_world(eef_pos, eef_quat_xyzw, q1=0.0, q2=0.0):
    R = quat_xyzw_to_mat(eef_quat_xyzw)
    p1 = np.asarray(eef_pos) + R @ (PAD1_OFFSET_E + PAD_SLIDE * q1)
    p2 = np.asarray(eef_pos) + R @ (PAD2_OFFSET_E + PAD_SLIDE * (-q2))
    return p1, p2


def cube_dist_to_point(center, half, point):
    """Distance from a point to an axis-aligned box (center, half-size)."""
    d = np.abs(np.asarray(point) - np.asarray(center)) - np.asarray(half)
    return float(np.linalg.norm(np.maximum(d, 0.0)))


# --- parse the log -----------------------------------------------------------

def parse_log():
    detail = []
    handover = []
    phases = []
    for raw in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("  detail case="):
            detail.append(line)
        elif line.startswith("HANDOVER case="):
            handover.append(line)
        elif re.match(r"case=\d+ t=\d+", line):
            phases.append(line)
    return detail, handover, phases


def kv(s: str) -> dict:
    """Very small JSON-ish parser for the log dict fragments."""
    s = s.replace("null", "null")
    try:
        return json.loads(s)
    except Exception:
        return {}


def parse_entry(line: str) -> dict:
    out = {}
    keys = ["eef0", "eef1", "q0", "q1", "payload", "trash", "targets",
            "payload_quat", "eef1_quat", "payload_frame_eef_offset"]
    positions = []
    for key in keys:
        m = re.search(rf"\b{key}=", line)
        if m:
            positions.append((m.start(), key))
    positions.sort()
    for i, (pos, key) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(line)
        seg = line[pos + len(key) + 1:end].strip()
        try:
            out[key] = json.loads(seg)
        except Exception:
            pass
    m = re.search(r"case=(\d+)", line)
    if m:
        out["case"] = int(m.group(1))
    m = re.search(r"t=(\d+)", line)
    if m:
        out["t"] = int(m.group(1))
    m = re.search(r"detail case=(\d+) t=(\d+)", line)
    if m:
        out["case"] = int(m.group(1))
        out["t"] = int(m.group(2))
    for side in ("left", "right"):
        m = re.search(rf"['\"]{side}['\"]:\s*([0-9.eE+-]+)", line)
        if m:
            out.setdefault("log_min_dist", {})[side] = float(m.group(1))
    return out


def main():
    detail, handover, phases = parse_log()
    print(f"log lines: detail={len(detail)} handover={len(handover)} phase={len(phases)}")

    # 1) chain regression at LID_GRASP entry (case 00 t=40, case 01 t=41)
    print("\n=== 1) chain regression: open-pad distances vs log min_dist ===")
    for line in detail:
        e = parse_entry(line)
        if e.get("t") not in (40, 41) or "q1" not in e:
            continue
        if e.get("case") != 0 and e.get("case") != 1:
            continue
        if "log_min_dist" not in e:
            continue
        p1, p2 = pad_world(e["eef1"], e["q1"], q1=0.04, q2=-0.04)
        trash = np.asarray(e["trash"])
        half = np.array([0.02, 0.02, 0.02])
        d1 = cube_dist_to_point(trash, half, p1)
        d2 = cube_dist_to_point(trash, half, p2)
        print(
            f"case {e['case']} t={e['t']}: computed pad-center dists "
            f"{d1:.4f}/{d2:.4f}  log min_dist {e['log_min_dist']['left']:.4f}/"
            f"{e['log_min_dist']['right']:.4f}"
        )

    # 2) closed pads vs cube: does the pinch bracket the cube?
    print("\n=== 2) closed pads (q=0) at LID_GRASP entry ===")
    for line in detail:
        e = parse_entry(line)
        if e.get("t") not in (40, 41) or "q1" not in e:
            continue
        if e.get("case") != 0 and e.get("case") != 1:
            continue
        p1, p2 = pad_world(e["eef1"], e["q1"], q1=0.0, q2=0.0)
        trash = np.asarray(e["trash"])
        half = np.array([0.02, 0.02, 0.02])
        d1 = cube_dist_to_point(trash, half, p1)
        d2 = cube_dist_to_point(trash, half, p2)
        R = quat_xyzw_to_mat(e["q1"])
        local_cube = R.T @ (trash - np.asarray(e["eef1"]))
        print(f"case {e['case']} t={e['t']}:")
        print(f"  closed pads at {p1} / {p2}")
        print(f"  pad-cube surface dists: {d1:.4f} / {d2:.4f}")
        print(f"  cube center in eef frame: {local_cube.round(4)}")
        print(f"  eef local x axis (world): {R[:,0].round(4)}")

    # 3) handover meeting-point geometry
    print("\n=== 3) handover meeting point vs payload ===")
    for line in handover:
        e = parse_entry(line)
        if e.get("t") not in (348, 349):
            continue
        payload = np.asarray(e["payload"])
        quat = np.asarray(e["payload_quat"])
        R = quat_xyzw_to_mat(quat)
        axis = R[:, 2]
        meeting = payload.copy()
        eef1 = np.asarray(e["eef1"])
        along = float(np.clip(np.dot(eef1 - meeting, axis), -0.09, 0.09))
        meeting += axis * along
        meeting[1] += 0.0
        meeting[2] = max(float(meeting[2]) + 0.055, 0.90)
        head_center = payload + R @ np.array([0.0, 0.0, 0.117767])
        print(f"case {e['case']} t={e['t']}:")
        print(f"  payload={payload} handle_axis={axis.round(4)}")
        print(f"  recomputed meeting target={meeting.round(5)}")
        print(f"  log target[1]={np.asarray(e['targets'])[1].round(5)}")
        print(f"  meeting z - payload z = {meeting[2]-payload[2]:.4f}")
        print(f"  head center={head_center.round(4)}")

    # 4) case 01 first-contact pad positions on the payload
    print("\n=== 4) case 01 TABLE_DESCEND first contact (t=330) ===")
    for line in handover:
        e = parse_entry(line)
        if e.get("t") != 330 or e.get("case") != 1:
            continue
        p1, p2 = pad_world(e["eef1"], e["eef1_quat"], q1=0.0, q2=0.0)
        payload = np.asarray(e["payload"])
        quat = np.asarray(e["payload_quat"])
        R = quat_xyzw_to_mat(quat)
        local_p1 = R.T @ (p1 - payload)
        local_p2 = R.T @ (p2 - payload)
        print(f"  pad1 world={p1.round(4)} payload-local={local_p1.round(4)}")
        print(f"  pad2 world={p2.round(4)} payload-local={local_p2.round(4)}")
        print(f"  handle spans local z in [-0.10, 0.10]; head center at +0.1178")
        print(f"  log payload_frame_eef_offset={np.asarray(e['payload_frame_eef_offset']).round(4)}")


if __name__ == "__main__":
    main()