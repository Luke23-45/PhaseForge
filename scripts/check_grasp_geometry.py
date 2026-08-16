"""Print the geometric constants that determine arm1's grasp feasibility:
  - Where the Panda gripper's eef site sits relative to the gripper base.
  - Where the left/right fingerpads sit relative to the eef site.
  - The hammer head box half-extents.
  - The hammer head's contact_geoms.
"""
import xml.etree.ElementTree as ET
import numpy as np

from robosuite.models.grippers import PandaGripper
from robosuite.utils.mjcf_utils import find_elements
from robosuite.models.objects.composite.hammer import HammerObject

print("=" * 60)
print("PANDA GRIPPER")
print("=" * 60)
g = PandaGripper()
root = g.root

# Find the eef site position in the gripper root frame.
eef_sites = find_elements(root=root, tags="site",
                          attribs={"name": "gripper0_eef"},
                          return_first=True)
if eef_sites is None:
    # try alternate name
    eef_sites = find_elements(root=root, tags="site",
                              attribs={"name": "gripper0_right_eef"},
                              return_first=True)
print("eef site element:", eef_sites)
if eef_sites is not None:
    print("  pos (in gripper root):", eef_sites.get("pos"))

# Walk the XML tree and collect every named element with a pos.
print("\nAll named elements with pos in gripper XML:")
for elem in root.iter():
    name = elem.get("name")
    pos = elem.get("pos")
    if name and pos and elem.tag in ("site", "body", "geom"):
        print(f"  {elem.tag:6s} {name:35s} pos={pos}")

print("\n--- Fingerpad geom details ---")
for pad_name in ("gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"):
    pad = find_elements(root=root, tags="geom",
                        attribs={"name": pad_name},
                        return_first=True)
    print(f"{pad_name}:")
    print(f"  pos: {pad.get('pos')}  size: {pad.get('size')}  type: {pad.get('type')}")

print()
print("=" * 60)
print("HAMMER HEAD")
print("=" * 60)
h = HammerObject(name="hammer")
hroot = h.root
for elem in hroot.iter():
    name = elem.get("name") or ""
    pos = elem.get("pos")
    size = elem.get("size")
    if elem.tag == "geom" and ("head" in name.lower() or pos is not None):
        print(f"  geom {name:30s} pos={pos} size={size} type={elem.get('type')}")

# Also check contact_geoms attribute on the hammer object.
print("\nHammerObject.contact_geoms:", getattr(h, "contact_geoms", "n/a"))
