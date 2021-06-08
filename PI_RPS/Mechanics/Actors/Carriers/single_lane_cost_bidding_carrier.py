"""
This is the carrier bidding its costs but only on the lane of the load's destination
"""
from PI_RPS.Mechanics.Actors.Carriers.carrier import CarrierWithCosts, SingleBidCarrier

from PI_RPS.prj_typing.types import CarrierBid
from typing import TYPE_CHECKING, Optional, List, Tuple

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Tools.load import Load
    from PI_RPS.Mechanics.Environment.environment import Environment


class SingleLaneCostBiddingCarrier(CarrierWithCosts, SingleBidCarrier):
    """Write description"""

    def __init__(self,
                 name: str,
                 home: 'Node',
                 in_transit: bool,
                 previous_node: 'Node',
                 next_node: 'Node',
                 time_to_go: int,
                 load: Optional['Load'],
                 environment: 'Environment',
                 episode_types: List[Tuple[str, 'Node', 'Node']],
                 episode_expenses: List[float],
                 episode_revenues: List[float],
                 this_episode_expenses: List[float],
                 this_episode_revenues: float,
                 transit_cost: float,
                 far_from_home_cost: float,
                 time_not_at_home: int,
                 max_time_not_at_home: int
                 ) -> None:

        super().__init__(name=name,
                         home=home,
                         in_transit=in_transit,
                         next_node=next_node,
                         previous_node=previous_node,
                         time_to_go=time_to_go,
                         load=load,
                         environment=environment,
                         episode_types=episode_types,
                         episode_expenses=episode_expenses,
                         episode_revenues=episode_revenues,
                         this_episode_expenses=this_episode_expenses,
                         this_episode_revenues=this_episode_revenues,
                         transit_cost=transit_cost,
                         far_from_home_cost=far_from_home_cost,
                         time_not_at_home=time_not_at_home)

        self._max_time_not_at_home: int = max_time_not_at_home

    def _decide_next_node(self) -> 'Node':
        """
        Go home only if more than self._max_time_not_at_home since last time at home
        """

        if self._time_not_at_home > self._max_time_not_at_home:
            return self._home
        else:
            return self._next_node

    def bid(self, next_node: 'Node') -> 'CarrierBid':
        """The bid function"""
        return self._calculate_costs(self._next_node, next_node)

    def _calculate_costs(self, from_node: 'Node', to_node: 'Node') -> float:
        result = 0.
        for delta_t in range(self._environment.get_distance(from_node, to_node)):
            t = self._time_not_at_home + delta_t
            result += self._transit_costs() + self._far_from_home_costs(time_not_at_home=t)
        return result
