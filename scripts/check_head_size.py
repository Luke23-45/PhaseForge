from robosuite.models.objects.composite.hammer import HammerObject
from robosuite.utils.mjcf_utils import find_elements

h = HammerObject(name='hammer')
root = h.root_body
for elem in root.iter():
    name = elem.get('name') or ''
    pos = elem.get('pos'); size = elem.get('size')
    if elem.tag == 'geom':
        print(f"  geom {name:35s} pos={pos} size={size} type={elem.get('type')} class={elem.get('class')}")
