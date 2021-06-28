"""
This is a carrier that bids the cost for each lane except if it has to go home. If so, it bids 0 on its home lane and
very high (higher than reserve price) on the other lanes and go home whatever the result
"""
import abc

from PI_RPS.Mechanics.Actors.Carriers.carrier import CarrierWithCosts, MultiBidCarrier, SingleBidCarrier

from typing import TYPE_CHECKING, Optional, List, Tuple

from PI_RPS.prj_typing.types import CarrierMultiBid, CarrierSingleBid, CostsTable, ListOfCostsTable

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Tools.load import Load
    from PI_RPS.Mechanics.Environment.environment import Environment


class LearningCostCarrier(CarrierWithCosts, abc.ABC):
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
                 episode_types: List[Tuple[str, 'Node', 'Node']],
                 episode_expenses: List[float],
                 episode_revenues: List[float],
                 this_episode_expenses: List[float],
                 this_episode_revenues: float,
                 transit_cost: float,
                 far_from_home_cost: float,
                 time_not_at_home: int,
                 nb_lost_auctions_in_a_row: int,
                 max_lost_auctions_in_a_row: int,
                 last_win_node: Node,
                 nb_history_since_last_win: int,
                 max_nb_infos_per_cost: int,
                 costs_table: Optional['CostsTable'],
                 list_of_costs_table: Optional['ListOfCostsTable'],
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

        self._nb_lost_auctions_in_a_row = nb_lost_auctions_in_a_row
        self._max_lost_auctions_in_a_row = max_lost_auctions_in_a_row

        self._last_win_node: 'Node' = last_win_node if last_win_node else self._next_node

        self._nb_history_since_last_win = nb_history_since_last_win
        self._max_nb_infos_per_cost: int = max_nb_infos_per_cost
        assert (costs_table is None and list_of_costs_table is None) or \
               (costs_table is not None and list_of_costs_table is not None), \
               "costs_table and list_of_costs_table should be both None or both assigned"
        if costs_table:
            self._costs_table: CostsTable = costs_table
            self._list_of_costs_table: ListOfCostsTable = list_of_costs_table
        else:
            self._costs_table: CostsTable = {}
            self._init_cost_tables()

        self._total_nb_cost_infos = 0
        self._total_max_nb_cost_infos = 0
        self._init_total_nb_cost_infos()

    def _init_cost_tables(self) -> None:
        self._costs_table = {}
        self._list_of_costs_table = {}
        for node1 in self._environment.nodes:
            self._costs_table[node1] = {}
            self._list_of_costs_table[node1] = {}
            for node2 in self._environment.nodes:
                if node2 != node1:
                    self._costs_table[node1][node2] = 0
                    self._list_of_costs_table[node1][node2] = [0]

    def _init_total_nb_cost_infos(self) -> None:
        for node1 in self._environment.nodes:
            for node2 in self._environment.nodes:
                if node2 != node1:
                    self._total_nb_cost_infos += len(self._list_of_costs_table[node1][node2])
                    self._total_max_nb_cost_infos += self._max_nb_infos_per_cost

    def _decide_next_node(self) -> 'Node':
        """
        Go home only if more than self._max_time_not_at_home since last time at home
        """
        if self._nb_lost_auctions_in_a_row > self._max_lost_auctions_in_a_row:
            return self._home
        else:
            return self._next_node

    def dont_get_attribution(self) -> None:
        super().dont_get_attribution()
        self._nb_lost_auctions_in_a_row += 1
        self._nb_history_since_last_win += 1

    def get_attribution(self, load: 'Load', next_node: 'Node', reserve_price_involved: bool) -> None:
        self._nb_history_since_last_win += 1
        # register the cost
        new_value = sum(self._episode_expenses[-self._nb_history_since_last_win:])
        costs_list = self._list_of_costs_table[self._last_win_node][next_node]
        costs_list.append(new_value)
        if len(self._list_of_costs_table[self._last_win_node][next_node]) > self._max_nb_infos_per_cost:
            old_value = costs_list.pop(0)
            self._costs_table[self._last_win_node][next_node] += (new_value - old_value) / self._max_nb_infos_per_cost
        else:
            self._costs_table[self._last_win_node][next_node] = sum(costs_list) / len(costs_list)
            self._total_nb_cost_infos += 1

        super().get_attribution(load, next_node, reserve_price_involved)
        self._nb_lost_auctions_in_a_row = 0
        self._last_win_node = next_node
        self._nb_history_since_last_win = 0

    @property
    def convergence_state(self):
        return self._total_nb_cost_infos / self._total_max_nb_cost_infos


class MultiLanesLearningCostCarrier(LearningCostCarrier, MultiBidCarrier):
    """
    It is a carrier but:
        * bidding their anticipated costs
        * is able to change its parameters
        * go back home when not seeing your boss (or mother) for a too long time
    """

    def bid(self) -> 'CarrierMultiBid':
        bid = {}
        for next_node in self._environment.nodes:
            if next_node != self._next_node:
                bid[next_node] = self._costs_table[self._next_node][next_node]
        return bid


class SingleLaneLearningCostCarrier(LearningCostCarrier, SingleBidCarrier):
    """
    It is a carrier but:
        * bidding their anticipated costs
        * is able to change its parameters
        * go back home when not seeing your boss (or mother) for a too long time
        * bid on the destination lane only
    """

    def bid(self, next_node: 'Node') -> 'CarrierSingleBid':
        """The bid function"""
        return self._costs_table[self._next_node][next_node]
