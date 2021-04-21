"""
The most basic carrier you could think of
"""
from random import random, expovariate
from Mechanics.Actors.carriers.carrier import Carrier, CarrierBid
from typing import TYPE_CHECKING, Optional, List
if TYPE_CHECKING:
    from Mechanics.Actors.nodes.node import Node
    from Mechanics.Tools.load import Load
    from Mechanics.environment import Environment


class DummyCarrier(Carrier):

    def __init__(self,
                 name: str,
                 home: 'Node',
                 in_transit: bool,
                 next_node: 'Node',
                 time_to_go: int,
                 load: Optional['Load'],
                 environment: 'Environment',
                 episode_expenses: List[float],
                 episode_revenues: List[float],
                 this_episode_expenses: List[float],
                 this_episode_revenues: float,
                 transit_cost: float,
                 far_from_home_cost: float):

        super().__init__(name,
                         home,
                         in_transit,
                         next_node,
                         time_to_go,
                         load,
                         environment,
                         episode_expenses,
                         episode_revenues,
                         this_episode_expenses,
                         this_episode_revenues)

        self._t_c: float = transit_cost
        self._ffh_c: float = far_from_home_cost

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

    def _transit_costs(self) -> float:
        """The transit costs"""
        return self._t_c

    def _far_from_home_costs(self) -> float:  # yes it is a constant, I told you it was dummy
        """The far from home costs"""
        return self._ffh_c

    def _update_ffh_cost_functions(self) -> None:
        """Here we do nothing"""
        pass
