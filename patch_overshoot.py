import re

path = r"C:\Users\Hellx\Documents\Programming\python\Project\Neryva\PhaseForge\phaseforge\evaluations\rollout\scripted_controller.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update HANDOVER_OVERSHOOT_Z
content = re.sub(
    r"HANDOVER_OVERSHOOT_Z: float = 0\.0\n",
    "HANDOVER_OVERSHOOT_Z: float = 0.055\n",
    content
)

# 2. Update TABLE_DESCEND grasp validation
table_descend_old = """            native_grasp = self._native_transport_grasp(1, "payload")
            if native_grasp is True:
                self._handover_native_stable_steps += 1"""

table_descend_new = """            native_grasp = self._native_transport_grasp(1, "payload")
            pad_contact = self._transport_pad_contact(1, "payload")
            if native_grasp is True or pad_contact is True:
                self._handover_native_stable_steps += 1"""

content = content.replace(table_descend_old, table_descend_new)

# 3. Update TABLE_RELEASE grasp validation
table_release_old = """            arm1_has_payload = self._native_transport_grasp(1, "payload")
            if arm1_has_payload is True:
                self._handover_native_stable_steps += 1"""

table_release_new = """            arm1_has_payload = self._native_transport_grasp(1, "payload")
            pad_contact = self._transport_pad_contact(1, "payload")
            if arm1_has_payload is True or pad_contact is True:
                self._handover_native_stable_steps += 1"""

content = content.replace(table_release_old, table_release_new)


with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Patched successfully.")
