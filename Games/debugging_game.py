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
cps2 = DummyCarrier('CParis2', ps, False, ps, 0, None, e, [], [], 3, 1)
cps3 = DummyCarrier('CParis3', ps, False, ps, 0, None, e, [], [], 3, 1)
cps4 = DummyCarrier('CParis4', ps, False, ps, 0, None, e, [], [], 3, 1)
cps5 = DummyCarrier('CParis5', ps, False, ps, 0, None, e, [], [], 3, 1)
cps6 = DummyCarrier('CParis6', ps, False, ps, 0, None, e, [], [], 3, 1)
cbx1 = DummyCarrier('CBrussels1', bx, False, bx, 0, None, e, [], [], 3, 1)
cbx2 = DummyCarrier('CBrussels2', bx, False, bx, 0, None, e, [], [], 3, 1)
cbx3 = DummyCarrier('CBrussels3', bx, False, bx, 0, None, e, [], [], 3, 1)
cbx4 = DummyCarrier('CBrussels4', bx, False, bx, 0, None, e, [], [], 3, 1)
cbx5 = DummyCarrier('CBrussels5', bx, False, bx, 0, None, e, [], [], 3, 1)
cbx6 = DummyCarrier('CBrussels6', bx, False, bx, 0, None, e, [], [], 3, 1)
chh1 = DummyCarrier('CHamburg1', hh, False, hh, 0, None, e, [], [], 3, 1)
chh2 = DummyCarrier('CHamburg2', hh, False, hh, 0, None, e, [], [], 3, 1)
chh3 = DummyCarrier('CHamburg3', hh, False, hh, 0, None, e, [], [], 3, 1)
chh4 = DummyCarrier('CHamburg4', hh, False, hh, 0, None, e, [], [], 3, 1)
chh5 = DummyCarrier('CHamburg5', hh, False, hh, 0, None, e, [], [], 3, 1)
chh6 = DummyCarrier('CHamburg6', hh, False, hh, 0, None, e, [], [], 3, 1)


for k in range(50):
    print(k)
    e.iteration()
print('end')
