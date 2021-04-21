"""
The most basic carrier you could think of
"""

from random import random, expovariate
from Mechanics.Actors.carriers.carrier import CarrierWithCosts

from typing import TYPE_CHECKING
from prj_typing.types import CarrierBid

if TYPE_CHECKING:
    from Mechanics.Actors.nodes.node import Node


class DummyCarrier(CarrierWithCosts):

    def bid(self, node: 'Node') -> CarrierBid:  # this should be a dictionary: key is next_node, value is float
        """To be called by the nodes before an auction"""
        bid = {}
        for next_node in self._environment.nodes:
            if next_node != node:
                bid[next_node] = expovariate(1 / 30)  # exponential law centered at 30 (told you it is dummy)
        return bid

    def _decide_next_node(self) -> 'Node':  # stay at same node with prob 0.9 or go home with prob 0.1
        """Decide of a next nodes after losing an auction (can be the same nodes when needed)"""
        home = random() < 0.1
        if home:
            return self._home
        else:
            return self._next_node
