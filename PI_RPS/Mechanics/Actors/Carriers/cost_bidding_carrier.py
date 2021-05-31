"""

"""

# TODO write description

import random

from PI_RPS.Mechanics.Actors.Carriers.carrier import CarrierWithCosts

from typing import TYPE_CHECKING, Optional, List

from PI_RPS.prj_typing.types import CarrierBid

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Tools.load import Load
    from PI_RPS.Mechanics.Environment.environment import Environment


class CostBiddingCarrier(CarrierWithCosts):
    """
    It is a carrier but:
        * bdding their anticipated costs
        * is able to change its parameters
        * go back home when not seeing your boss (or mother) for a too long time
    """

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
                 far_from_home_cost: float,
                 t_c_mu: float,
                 t_c_sigma: float,
                 ffh_c_mu: float,
                 ffh_c_sigma: float,
                 too_high_bid: float,
                 time_not_at_home: int,
                 max_time_not_at_home: int
                 ) -> None:

        super().__init__(name=name,
                         home=home,
                         in_transit=in_transit,
                         next_node=next_node,
                         time_to_go=time_to_go,
                         load=load,
                         environment=environment,
                         episode_expenses=episode_expenses,
                         episode_revenues=episode_revenues,
                         this_episode_expenses=this_episode_expenses,
                         this_episode_revenues=this_episode_revenues,
                         transit_cost=transit_cost,
                         far_from_home_cost=far_from_home_cost,
                         time_not_at_home=time_not_at_home)

        self._t_c_mu: float = t_c_mu
        self._t_c_sigma: float = t_c_sigma
        self._ffh_c_mu: float = ffh_c_mu
        self._ffh_c_sigma: float = ffh_c_sigma
        self._too_high_bid: float = too_high_bid  # should be at least bigger than the transformed reserve price
        # of the shippers
        self._max_time_not_at_home: int = max_time_not_at_home

    def _decide_next_node(self) -> 'Node':
        """
        Go home only if more than self._max_time_not_at_home since last time at home
        """

        if self._time_not_at_home > self._max_time_not_at_home:
            return self._home
        else:
            return self._next_node

    def bid(self, node: 'Node') -> 'CarrierBid':
        bid = {}
        if self._time_not_at_home > self._max_time_not_at_home:
            for next_node in self._environment.nodes:
                if next_node != node:
                    bid[next_node] = 0 if next_node == self._home else self._too_high_bid
        else:
            for next_node in self._environment.nodes:
                if next_node != node:
                    bid[next_node] = self._calculate_costs(node, next_node)

        return bid

    def _calculate_costs(self, from_node: 'Node', to_node: 'Node') -> float:
        result = 0.
        for delta_t in range(self._environment.get_distance(from_node, to_node)):
            t = self._time_not_at_home + delta_t
            result += self._transit_costs() + self._far_from_home_costs(time_not_at_home=t)
        return result

    def random_new_cost_parameters(self) -> None:
        road_costs = random.normalvariate(mu=self._t_c_mu, sigma=self._t_c_sigma)
        drivers_costs = random.normalvariate(mu=self._ffh_c_mu, sigma=self._ffh_c_sigma)
        self._set_new_cost_parameters(t_c=road_costs, ffh_c=drivers_costs)

    def _set_new_cost_parameters(self, t_c: float, ffh_c: float) -> None:
        """
        Setting new parameters for learners and resetting buffers
        """
        self._t_c = t_c
        self._ffh_c = ffh_c
        self.clear_profit()
