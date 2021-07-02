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


class LearningCostsCarrier(CarrierWithCosts, abc.ABC):
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
                 last_won_node: Optional['Node'],
                 nb_episode_at_last_won_node: int,
                 nb_lives: int,
                 max_nb_infos_per_node: int,
                 costs_table: Optional['CostsTable'],
                 list_of_costs_table: Optional['ListOfCostsTable'],
                 is_learning: bool,
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

        self._nb_lives = nb_lives

        self._is_learning = is_learning

        self._nb_lost_auctions_in_a_row = nb_lost_auctions_in_a_row
        self._max_lost_auctions_in_a_row = max_lost_auctions_in_a_row

        self._last_won_node: Optional['Node'] = last_won_node
        self._nb_episode_at_last_won_node: int = nb_episode_at_last_won_node

        self._max_nb_infos_per_node: int = max_nb_infos_per_node
        assert (costs_table is None and list_of_costs_table is None) or \
               (costs_table is not None and list_of_costs_table is not None), \
            "costs_table and list_of_costs_table should be both None or both assigned"
        if costs_table:
            self._costs_table: CostsTable = costs_table
            self._list_of_costs_table: ListOfCostsTable = list_of_costs_table
        else:
            self._costs_table: CostsTable = {}
            self._list_of_costs_table = {}
            self._init_cost_tables()

        self._total_nb_cost_infos = 0
        self._total_max_nb_cost_infos = 0
        self._nb_cost_infos = {}
        self._init_total_nb_cost_infos()

    def _init_cost_tables(self) -> None:
        assert len(self._costs_table.keys()) == 0, 'Init only empty costs_tables'
        assert len(self._list_of_costs_table.keys()) == 0, 'Init only empty list_of_costs_table'
        for node1 in self._environment.nodes:
            self._costs_table[node1] = 0
            self._list_of_costs_table[node1] = []

    def _init_total_nb_cost_infos(self) -> None:
        assert self._total_nb_cost_infos == 0, 'Only init if 0'
        assert self._total_max_nb_cost_infos == 0, 'Only init if 0'
        for node1 in self._environment.nodes:
            length = len(self._list_of_costs_table[node1])
            self._nb_cost_infos[node1] = length
            self._total_nb_cost_infos += length
            self._total_max_nb_cost_infos += self._max_nb_infos_per_node

    def reinit_cost_tables_to_average(self):
        for node1 in self._environment.nodes:
            self._list_of_costs_table[node1] = [self._costs_table[node1]]
        self._total_nb_cost_infos = 0
        self._total_max_nb_cost_infos = 0
        self._init_total_nb_cost_infos()

    def reinit_cost_tables_to_0(self):
        for node1 in self._environment.nodes:
            self._costs_table[node1] = 0
            self._list_of_costs_table[node1] = []
        self._total_nb_cost_infos = 0
        self._total_max_nb_cost_infos = 0
        self._init_total_nb_cost_infos()

    def remove_a_life(self):
        if self._nb_lives == 0:
            self._delete()
        else:
            self._nb_lives -= 1

    def _delete(self):
        # loads and auction will be cleaned, so we need to get out of the actors and environment only, not the tools
        if not self._in_transit:
            self._next_node.remove_carrier_from_waiting_list(self)
        self._environment.remove_carrier(self)

    def _decide_next_node(self) -> 'Node':
        """
        Go home only if more than self._max_time_not_at_home since last time at home
        """
        if self._nb_lost_auctions_in_a_row > self._max_lost_auctions_in_a_row:
            return self._home
        else:
            return self._next_node

    def dont_get_attribution(self) -> None:
        self._nb_lost_auctions_in_a_row += 1
        super().dont_get_attribution()

    def get_attribution(self, load: 'Load', next_node: 'Node', reserve_price_involved: bool) -> None:
        super().get_attribution(load, next_node, reserve_price_involved)
        # register value
        if self._is_learning and (self._last_won_node is not None):
            new_value = sum(self._episode_expenses[-self._nb_episode_at_last_won_node:]) \
                if self._nb_episode_at_last_won_node > 0 else 0.
            node_info_list = self._list_of_costs_table[self._last_won_node]
            node_info_list.append(new_value)
            if len(node_info_list) > self._max_nb_infos_per_node:
                old_value = node_info_list.pop(0)
                self._costs_table[self._last_won_node] += (new_value - old_value) / self._max_nb_infos_per_node
            else:
                self._costs_table[self._last_won_node] = sum(node_info_list) / len(node_info_list)
                self._total_nb_cost_infos += 1
                self._nb_cost_infos[self._last_won_node] += 1
        # prepare for next round
        self._nb_lost_auctions_in_a_row = 0
        # prepare for next round
        self._last_won_node = next_node
        self._nb_episode_at_last_won_node = 0

    def next_step(self) -> None:
        if not self._in_transit:
            self._nb_episode_at_last_won_node += 1

    def _calculate_costs(self, from_node: 'Node', to_node: 'Node') -> float:
        """Will be called by bid"""
        result = 0.
        for delta_t in range(self._environment.get_distance(from_node, to_node)):
            t = self._time_not_at_home + delta_t
            result += self._transit_costs() + self._far_from_home_costs(time_not_at_home=t)
        return result

    def convergence_state(self):
        return self._total_nb_cost_infos / self._total_max_nb_cost_infos

    @property
    def is_learning(self) -> bool:
        return self._is_learning

    @is_learning.setter
    def is_learning(self, value) -> None:
        assert type(value) == bool, "value is not a bool"
        assert value == (not self._is_learning), "Only change to opposite"
        self._is_learning = value

    @property
    def max_lost_auctions_in_a_row(self) -> int:
        return self._max_lost_auctions_in_a_row

    @property
    def max_nb_infos_per_node(self) -> int:
        return self._max_nb_infos_per_node

    @property
    def nb_lives(self) -> int:
        return self._nb_lives


class MultiLanesLearningCostsCarrier(LearningCostsCarrier, MultiBidCarrier):
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
                bid[next_node] = self._costs_table[next_node] + self._calculate_costs(self._next_node, next_node)
        return bid


class SingleLaneLearningCostsCarrier(LearningCostsCarrier, SingleBidCarrier):
    """
    It is a carrier but:
        * bidding their anticipated costs
        * is able to change its parameters
        * go back home when not seeing your boss (or mother) for a too long time
        * bid on the destination lane only
    """

    def bid(self, next_node: 'Node') -> 'CarrierSingleBid':
        """The bid function"""
        return self._costs_table[next_node] + self._calculate_costs(self._next_node, next_node)
