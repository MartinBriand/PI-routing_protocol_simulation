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

spshh = DummyShipper('Paris->Hamburg', [NodeLaw(ps, hh, lambda: 1, {})], [], [], e)
shhps = DummyShipper('Hamburg->Paris', [NodeLaw(hh, ps, lambda: 1, {})], [], [], e)

cps1 = DummyCarrier('CParis1', ps, False, ps, 0, None, e, [], [], 3, 1)
# cps2 = DummyCarrier('CParis2', ps, False, ps, 0, None, e, [], [], 3, 1)
# cps3 = DummyCarrier('CParis3', ps, False, ps, 0, None, e, [], [], 3, 1)
cbx = DummyCarrier('CBrussels', bx, False, bx, 0, None, e, [], [], 3, 1)
chh = DummyCarrier('CHamburg', hh, False, hh, 0, None, e, [], [], 3, 1)


for k in range(10):
    print(k)
    e.iteration()
print('end')
