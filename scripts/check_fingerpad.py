import robosuite, numpy as np
from robosuite.models.grippers import PandaGripper
from robosuite.utils.mjcf_utils import find_elements
g = PandaGripper()
pads = find_elements(root=g.root, tags='geom', attribs={'name': 'gripper0_finger1_pad_collision'}, return_first=True)
print('left fingerpad pos:', pads.get('pos'), 'size:', pads.get('size'))
pads2 = find_elements(root=g.root, tags='geom', attribs={'name': 'gripper0_finger2_pad_collision'}, return_first=True)
print('right fingerpad pos:', pads2.get('pos'), 'size:', pads2.get('size'))
for elem in g.root.iter():
    name = elem.get('name') or ''
    if 'finger' in name and elem.tag in ('body', 'joint'):
        pos = elem.get('pos'); axis = elem.get('axis'); rng = elem.get('range')
        print(f'{elem.tag} {name}: pos={pos} axis={axis} range={rng}')
