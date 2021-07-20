"""
This is a carrier that bids the cost for each lane times a cost majoration to have positive profit.
"""
import abc

from PI_RPS.Mechanics.Actors.Carriers.carrier import CarrierWithCosts, MultiBidCarrier, SingleBidCarrier

from typing import TYPE_CHECKING, Optional, List, Tuple

from PI_RPS.prj_typing.types import CarrierMultiBid, CarrierSingleBid

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Tools.load import Load
    from PI_RPS.Mechanics.Environment.environment import Environment


class CostBiddingCarrier(CarrierWithCosts, abc.ABC):
    """
    Base abstract class for Cost bidding carriers
    """

    def __init__(self,
                 name: str,
                 home: 'Node',
                 in_transit: bool,
                 previous_node: 'Node',
                 next_node: 'Node',
                 time_to_go: int,
                 load: Optional['Load'],
                 environment: 'Environment',
                 episode_types: List[Tuple[str, 'Node', 'Node', bool]],
                 episode_expenses: List[float],
                 episode_revenues: List[float],
                 this_episode_expenses: List[float],
                 this_episode_revenues: float,
                 transit_cost: float,
                 far_from_home_cost: float,
                 time_not_at_home: int,
                 max_time_not_at_home: int,
                 nb_lost_auctions_in_a_row: int,
                 max_lost_auctions_in_a_row: int,
                 cost_majoration: float
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

        self._nb_lost_auctions_in_a_row: int = nb_lost_auctions_in_a_row
        self._max_lost_auctions_in_a_row: int = max_lost_auctions_in_a_row
        self._max_time_not_at_home: int = max_time_not_at_home
        self._cost_majoration: float = cost_majoration

    def _decide_next_node(self) -> 'Node':
        """
        Go home only if more than self._max_time_not_at_home since last time at home or if loosing too much
        """
        if self._nb_lost_auctions_in_a_row > self._max_lost_auctions_in_a_row or \
                self._time_not_at_home > self._max_time_not_at_home:
            return self._home
        else:
            return self._next_node

    def dont_get_attribution(self) -> None:
        self._nb_lost_auctions_in_a_row += 1
        super().dont_get_attribution()

    def get_attribution(self, load: 'Load', next_node: 'Node', reserve_price_involved: bool) -> None:
        super().get_attribution(load, next_node, reserve_price_involved)
        self._nb_lost_auctions_in_a_row = 0

    def _calculate_majored_costs(self, from_node: 'Node', to_node: 'Node') -> float:
        """Will be called by bid"""
        result = 0.
        for delta_t in range(self._environment.get_distance(from_node, to_node)):
            t = self._time_not_at_home + delta_t
            result += self._transit_costs() + self._far_from_home_costs(time_not_at_home=t)
        return result * self._cost_majoration

    @property
    def cost_majoration(self) -> float:
        return self._cost_majoration


class MultiLanesCostBiddingCarrier(CostBiddingCarrier, MultiBidCarrier):
    """
    MultiLanes version
    """

    def bid(self) -> 'CarrierMultiBid':
        bid = {}
        for next_node in self._environment.nodes:
            if next_node != self._next_node:
                bid[next_node] = self._calculate_majored_costs(self._next_node, next_node)
        return bid


class SingleLaneCostBiddingCarrier(CostBiddingCarrier, SingleBidCarrier):
    """
    SingleLane version
    """

    def bid(self, next_node: 'Node') -> 'CarrierSingleBid':
        """The bid function"""
        return self._calculate_majored_costs(self._next_node, next_node)
