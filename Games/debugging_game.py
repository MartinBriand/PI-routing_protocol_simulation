"""
This script is meant to be changed and tun games to debug the structure and mechanics of the game
It is supposed to:
    * instantiate everything
    * launch visualizers
    * run an iteration loop on the environment
"""

from Mechanics.Environment.environment import Environment
from Mechanics.Actors.nodes.dummy_node import DummyNode
from Mechanics.Actors.carriers.dummy_carrier import DummyCarrier
from Mechanics.Actors.shippers.dummy_shipper import DummyShipper
from Mechanics.Actors.shippers.shipper import NodeLaw

e = Environment()

ps = DummyNode('Paris', {}, 100, [], e)
bx = DummyNode('Brussels', {}, 100, [], e)
hh = DummyNode('Hamburg', {}, 100, [], e)

for node in e.nodes:
    node.initialize_weights()
del node

distances = {ps: {bx: 3, hh: 6}, bx: {ps: 3, hh: 4}, hh: {ps: 6, bx: 4}}
e.set_distance(distances)
del distances

DummyShipper('Paris->Hamburg', [NodeLaw(ps, hh, lambda: 1, {})], [], [], e)
DummyShipper('Hamburg->Paris', [NodeLaw(hh, ps, lambda: 1, {})], [], [], e)

for k in range(10):
    DummyCarrier(name='CParis_{}'.format(k),
                 home=ps,
                 in_transit=False,
                 next_node=ps,
                 time_to_go=0,
                 load=None,
                 environment=e,
                 episode_expenses=[],
                 episode_revenues=[],
                 this_episode_expenses=[],
                 this_episode_revenues=0,
                 transit_cost=3,
                 far_from_home_cost=1,
                 time_not_at_home=0)
    DummyCarrier(name='CBrussels_{}'.format(k),
                 home=bx,
                 in_transit=False,
                 next_node=bx,
                 time_to_go=0,
                 load=None,
                 environment=e,
                 episode_expenses=[],
                 episode_revenues=[],
                 this_episode_expenses=[],
                 this_episode_revenues=0,
                 transit_cost=3,
                 far_from_home_cost=1,
                 time_not_at_home=0)
    DummyCarrier(name='CHamburg_{}'.format(k),
                 home=hh,
                 in_transit=False,
                 next_node=hh,
                 time_to_go=0,
                 load=None,
                 environment=e,
                 episode_expenses=[],
                 episode_revenues=[],
                 this_episode_expenses=[],
                 this_episode_revenues=0,
                 transit_cost=3,
                 far_from_home_cost=1,
                 time_not_at_home=0)

del ps, bx, hh

for k in range(1000):
    e.iteration()
print('end')
