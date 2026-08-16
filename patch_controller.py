import re

path = r"C:\Users\Hellx\Documents\Programming\python\Project\Neryva\PhaseForge\phaseforge\evaluations\rollout\scripted_controller.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update TRASH_GRASP_Z_OFFSET to 0.005
content = re.sub(
    r"TRASH_GRASP_Z_OFFSET: float = 0.01",
    "TRASH_GRASP_Z_OFFSET: float = 0.005",
    content
)

# 2. Update HANDOVER_OVERSHOOT_Z to 0.06
content = re.sub(
    r"HANDOVER_OVERSHOOT_Z: float = 0.07",
    "HANDOVER_OVERSHOOT_Z: float = 0.06",
    content
)

# 3. Add TRASH_RELEASE to _TransportPhase
content = re.sub(
    r"    TRASH_PLACE = auto\(\)\n    # --- Mid-Air",
    "    TRASH_PLACE = auto()\n    TRASH_RELEASE = auto()\n    # --- Mid-Air",
    content
)

# 4. Update TRASH_PLACE and add TRASH_RELEASE logic
trash_place_old = """            if (reached or is_stalled) and t - self._transport_started_at >= self.config.place_hold_steps:
                self._transport_phase = _TransportPhase.TABLE_TRANSPORT
                self._stall_since = None
                self._last_target = None
                return self._two_arm_action(targets, eef0, eef1, (GRIPPER_CLOSE, GRIPPER_OPEN))
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_CLOSE, GRIPPER_CLOSE))"""

trash_place_new = """            if (reached or is_stalled) and t - self._transport_started_at >= self.config.place_hold_steps:
                self._transport_phase = _TransportPhase.TRASH_RELEASE
                self._transport_started_at = t
                self._stall_since = None
                self._last_target = None
                return self._two_arm_action(targets, eef0, eef1, (GRIPPER_CLOSE, GRIPPER_OPEN))
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_CLOSE, GRIPPER_CLOSE))

        if self._transport_phase is _TransportPhase.TRASH_RELEASE:
            targets = self._place_targets
            self._transport_watchdog(targets, eef0, eef1, t)
            assert self._transport_started_at is not None
            if t - self._transport_started_at >= 15:
                self._transport_phase = _TransportPhase.TABLE_TRANSPORT
                self._stall_since = None
                self._last_target = None
            return self._two_arm_action(targets, eef0, eef1, (GRIPPER_CLOSE, GRIPPER_OPEN))"""

content = content.replace(trash_place_old, trash_place_new)


# 5. Fix TABLE_RELEASE so it actually verifies arm1 grasp
table_release_old = """            assert self._transport_started_at is not None
            if t - self._transport_started_at >= self.config.grasp_hold_steps:
                self._transport_phase = _TransportPhase.TABLE_RETRACT
                self._transport_started_at = t
                self._stall_since = None
                self._last_target = None
                self._handover_arm1_snapshot = arm1_hold.copy()
            return self._two_arm_action(
                targets, eef0, eef1,
                (GRIPPER_CLOSE, GRIPPER_CLOSE),
                velocity_scales=(swing_scale, swing_scale),
            )"""

table_release_new = """            assert self._transport_started_at is not None
            held_for = t - self._transport_started_at
            if held_for >= self.config.grasp_hold_steps:
                arm1_has_payload = self._native_transport_grasp(1, "payload")
                arm1_pad = self._transport_pad_contact(1, "payload")
                if arm1_has_payload is not False or arm1_pad is not False:
                    self._transport_phase = _TransportPhase.TABLE_RETRACT
                    self._transport_started_at = t
                    self._stall_since = None
                    self._last_target = None
                    self._handover_arm1_snapshot = arm1_hold.copy()
                elif held_for >= self.config.grasp_hold_steps + self.GRASP_CONFIRM_STEPS:
                    self._stalled_from_phase = self._transport_phase.name
                    self._transport_phase = _TransportPhase.STALLED
            return self._two_arm_action(
                targets, eef0, eef1,
                (GRIPPER_CLOSE, GRIPPER_CLOSE),
                velocity_scales=(swing_scale, swing_scale),
            )"""

content = content.replace(table_release_old, table_release_new)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Patched scripted_controller.py successfully.")
