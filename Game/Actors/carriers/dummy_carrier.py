"""
The most basic carrier you could think of
"""
from random import random, randint

from Game.Actors.carriers.carrier import Carrier


class DummyCarrier(Carrier):

    def __init__(self, name, home, in_transit, next_node, time_to_go, load, environment, expenses, revenues,
                 transit_cost, far_from_home_cost):
        super.__init__(name, home, in_transit, next_node, time_to_go, load, environment, expenses, revenues)
        self.transit_costs = transit_cost
        self.far_from_home_costs = far_from_home_cost

    def bid(self, node):  # this should be a dictionary: key is next_node, value is float
        # TODO: implement this method
        """To be called by the nodes before an auction"""
        raise NotImplementedError

    def _decide_next_node(self):  # stay at same node with prob 0.9 or go home with prob 0.1
        """Decide of a next nodes after losing an auction (can be the same nodes when needed)"""
        leave = random() < 0.1
        if leave:
            nodes = self.environment.nodes
            return nodes[randint(0, len(nodes) - 1)]
        else:
            return self.next_node

    def _transit_costs(self):
        return self.transit_costs

    def _far_from_home_costs(self):
        return self.far_from_home_costs
