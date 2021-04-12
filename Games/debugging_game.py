"""The game, properly speaking... aka the script that we launch the game and the visualizers"""

# instantiate everything
# launch visualizers
# run a iteration while loop on the environment

from Mechanics.environment import Environment
from Mechanics.Actors.nodes.dummy_node import DummyNode
from Mechanics.Actors.carriers.dummy_carrier import DummyCarrier
from Mechanics.Actors.shippers.dummy_shipper import DummyShipper
from Mechanics.Actors.shippers.shipper import NodeLaw

e = Environment()

ps = DummyNode('Paris', {}, {}, [], e)
bx = DummyNode('Brussels', {}, {}, [], e)
hh = DummyNode('Hamburg', {}, {}, [], e)

for node in e.nodes:
    node.initialize_weights()

distances = {ps: {bx: 3, hh: 6}, bx: {ps: 3, hh: 4}, hh: {ps: 6, bx: 4}}
e.set_distance(distances)

DummyShipper('Paris->Hamburg', [NodeLaw(ps, hh, lambda: 1, {})], [], [], e)
DummyShipper('Hamburg->Paris', [NodeLaw(hh, ps, lambda: 1, {})], [], [], e)

for k in range(10):
    DummyCarrier('CParis_{}'.format(k), ps, False, ps, 0, None, e, [], [], 3, 1)
    DummyCarrier('CBrussels_{}'.format(k), bx, False, bx, 0, None, e, [], [], 3, 1)
    DummyCarrier('CHamburg_{}'.format(k), hh, False, hh, 0, None, e, [], [], 3, 1)


for k in range(100000):
    e.iteration()
print('end')
