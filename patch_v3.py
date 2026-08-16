"""
Two targeted patches:

1. TABLE_DESCEND: arm1 currently just holds at the frozen meeting_target after TABLE_TRANSPORT
   converges at d1<=0.12m. That puts arm1 up to 12cm from the handle with no active press.
   Fix: In TABLE_DESCEND, when grasp_offset_local is not yet set, drive arm1 actively toward
   payload body center (with a small Z overshoot so pads actually penetrate the handle),
   instead of holding at the frozen 12-cm-away meeting_target.

2. LID_GRASP (case 00): When arm1_trash stays False for the entire hold period AND 
   pad_contact is also False, the code waits for GRASP_CONFIRM_STEPS before stalling.
   But the real problem is that arm1's trash grasp target (`trash_grasp`) is computed
   from the state at LID_DESCEND entry and may be slightly off for some reset seeds.
   Fix: If trash_grasped is False after grasp_hold_steps, re-drive arm1 slightly lower
   by introducing a TRASH_DESCEND_RECOVERY offset so arm1 can "find" the trash.
"""

import re

path = r"C:\Users\Hellx\Documents\Programming\python\Project\Neryva\PhaseForge\phaseforge\evaluations\rollout\scripted_controller.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# ─────────────────────────────────────────────────────────────────────────────
# Patch 1: TABLE_DESCEND — actively drive arm1 toward payload until contact
# Replace the arm1_hold logic: when no grasp_offset_local yet, target payload
# body center + small overshoot so pads physically reach the handle.
# ─────────────────────────────────────────────────────────────────────────────

old_td = """            arm0_hold = eef0.copy()
            arm1_hold = (
                self._handover_arm1_grasp_target.copy()
                if self._handover_arm1_grasp_target is not None
                else (
                    self._handover_meeting_target.copy()
                    if self._handover_meeting_target is not None
                    else meeting
                )
            )
            targets = (arm0_hold, arm1_hold)
            self._transport_watchdog(targets, eef0, eef1, t)"""

new_td = """            arm0_hold = eef0.copy()
            if self._handover_arm1_grasp_target is not None:
                # Contact is established; track payload frame to not lose pads.
                arm1_hold = self._handover_arm1_grasp_target.copy()
            elif self._handover_arm1_grasp_offset_local is None:
                # No contact yet: actively press arm1 toward the payload handle.
                # Use the projected handle point but pull the Z target DOWN to
                # payload_z + a tiny overshoot so the fingertips physically
                # penetrate the handle surface and register contact.
                base = (
                    self._handover_meeting_target.copy()
                    if self._handover_meeting_target is not None
                    else meeting.copy()
                )
                # Drive wrist to payload_z + OVERSHOOT_Z (fingertips hit handle)
                base[2] = payload[2] + self.HANDOVER_OVERSHOOT_Z
                base[2] = max(base[2], self.HANDOVER_Z_MIN)
                arm1_hold = base
            else:
                arm1_hold = (
                    self._handover_meeting_target.copy()
                    if self._handover_meeting_target is not None
                    else meeting.copy()
                )
            targets = (arm0_hold, arm1_hold)
            self._transport_watchdog(targets, eef0, eef1, t)"""

content = content.replace(old_td, new_td)

# ─────────────────────────────────────────────────────────────────────────────
# Patch 2: LID_GRASP trash recovery — if arm1_trash native grasp fails after
# hold steps, incrementally descend arm1 by re-computing trash_grasp with a
# growing negative Z offset (gentle probe downward) instead of immediately
# stalling. We do this by switching to an active re-descend when trash is False.
# ─────────────────────────────────────────────────────────────────────────────

old_lid_grasp = """        if self._transport_phase is _TransportPhase.LID_GRASP:
            assert self._transport_started_at is not None
            held_for = t - self._transport_started_at
            if held_for >= self.config.grasp_hold_steps:
                lid_grasped = self._native_transport_grasp(0, "lid")
                trash_grasped = self._native_transport_grasp(1, "trash")
                if lid_grasped is not False and trash_grasped is not False:
                    self._transport_phase = _TransportPhase.LID_LIFT
                    self._stall_since = None
                    self._last_target = None
                elif (
                    lid_grasped is False or trash_grasped is False
                ) and held_for >= self.config.grasp_hold_steps + self.GRASP_CONFIRM_STEPS:
                    self._stalled_from_phase = self._transport_phase.name
                    self._transport_phase = _TransportPhase.STALLED
            return self._two_arm_action(
                (eef0, eef1), eef0, eef1, (GRIPPER_CLOSE, GRIPPER_CLOSE)
            )"""

new_lid_grasp = """        if self._transport_phase is _TransportPhase.LID_GRASP:
            assert self._transport_started_at is not None
            held_for = t - self._transport_started_at
            lid_grasped = self._native_transport_grasp(0, "lid")
            trash_grasped = self._native_transport_grasp(1, "trash")
            trash_pad = self._transport_pad_contact(1, "trash")
            if held_for >= self.config.grasp_hold_steps:
                if lid_grasped is not False and (trash_grasped is not False or trash_pad is not False):
                    self._transport_phase = _TransportPhase.LID_LIFT
                    self._stall_since = None
                    self._last_target = None
                elif (
                    lid_grasped is False or (trash_grasped is False and trash_pad is False)
                ) and held_for >= self.config.grasp_hold_steps + self.GRASP_CONFIRM_STEPS:
                    self._stalled_from_phase = self._transport_phase.name
                    self._transport_phase = _TransportPhase.STALLED
            # If arm1 has not made trash contact yet, probe deeper.
            # Each step past grasp_hold_steps lower the arm1 target by 1 mm (capped at 15 mm).
            if trash_grasped is False and trash_pad is False and held_for >= self.config.grasp_hold_steps:
                probe_steps = min(held_for - self.config.grasp_hold_steps, 15)
                probe_target = trash_grasp.copy()
                probe_target[2] -= probe_steps * 0.001
                return self._two_arm_action(
                    (eef0, probe_target), eef0, eef1, (GRIPPER_CLOSE, GRIPPER_CLOSE)
                )
            return self._two_arm_action(
                (eef0, eef1), eef0, eef1, (GRIPPER_CLOSE, GRIPPER_CLOSE)
            )"""

content = content.replace(old_lid_grasp, new_lid_grasp)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Patched successfully.")
print()

# Verify patches applied
if "actively press arm1 toward the payload handle" in content:
    print("✓ Patch 1 (TABLE_DESCEND active press) applied")
else:
    print("✗ Patch 1 FAILED")

if "probe deeper" in content:
    print("✓ Patch 2 (LID_GRASP trash probe) applied")
else:
    print("✗ Patch 2 FAILED")
