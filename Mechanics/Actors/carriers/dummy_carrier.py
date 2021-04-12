"""
The most basic carrier you could think of
"""
from random import random, randint, expovariate

from Mechanics.Actors.carriers.carrier import Carrier


class DummyCarrier(Carrier):

    def __init__(self, name, home, in_transit, next_node, time_to_go, load, environment, expenses, revenues,
                 transit_cost, far_from_home_cost):
        super().__init__(name, home, in_transit, next_node, time_to_go, load, environment, expenses, revenues)
        self.transit_costs = transit_cost
        self.far_from_home_costs = far_from_home_cost

    def bid(self, node):  # this should be a dictionary: key is next_node, value is float
        """To be called by the nodes before an auction"""
        bid = {}
        for next_node in self.environment.nodes:
            if next_node != node:
                bid[next_node] = expovariate(1 / 30)  # exponential law centered at 30 (told you it is dummy)
        return bid

    def _decide_next_node(self):  # stay at same node with prob 0.9 or go home with prob 0.1
        """Decide of a next nodes after losing an auction (can be the same nodes when needed)"""
        home = random() < 0.1
        if home:
            return self.home
        else:
            return self.next_node

    def _transit_costs(self):
        """The transit costs"""
        return self.transit_costs

    def _far_from_home_costs(self):  # yes it is a constant, I told you it was dummy
        """The far from home costs"""
        return self.far_from_home_costs
