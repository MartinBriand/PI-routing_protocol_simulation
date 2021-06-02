"""
Carrier file
"""
import abc
from typing import TYPE_CHECKING, Optional, List
from math import exp

from PI_RPS.prj_typing.types import CarrierBid

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Tools.load import Load
    from PI_RPS.Mechanics.Environment.environment import Environment


class Carrier(abc.ABC):
    """
    A Carriers has two states:
        * it is either in transit, if so, it is forced to finish it journey
        * If not, it is because it is at a Nodes, where, it can participate in an auction, and depending on the result
            * will either be attributed a good to transport and will have to carry it to the next Nodes
            * Or will not get a good and may decide to stay or move to another Nodes
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
                 this_episode_revenues: float) -> None:

        self._name: str = name
        self._home: 'Node' = home
        # state: if not in transit, we are at Nodes next_node, ef not time_to_go > 0 and we are going to next_node
        self._in_transit: bool = in_transit
        self._next_node: 'Node' = next_node  # instantiated after the creation of the node
        self._time_to_go: int = time_to_go
        self._load: 'Load' = load
        self._environment: 'Environment' = environment  # instantiated after the creation of the environment

        # costs are allowed to be methods

        self._episode_expenses: List[float] = episode_expenses
        self._episode_revenues: List[float] = episode_revenues
        self._this_episode_expenses: List[float] = this_episode_expenses
        self._this_episode_revenues: float = this_episode_revenues
        self._total_expenses: float = sum(self._episode_expenses) + sum(self._this_episode_expenses)
        self._total_revenues: float = sum(self._episode_revenues) + self._this_episode_revenues

        self._environment.add_carrier(self)

        if not self._in_transit:  # should be instantiated after the creations of the Nodes
            self._next_node.add_carrier_to_waiting_list(self)

    @abc.abstractmethod
    def bid(self, node: 'Node') -> CarrierBid:
        """To be called by the Nodes before an auction"""

    @abc.abstractmethod
    def _decide_next_node(self) -> 'Node':
        """Decide of a next Nodes after losing an auction (can be the same Nodes when needed)"""

    def get_attribution(self, load: 'Load', next_node: 'Node') -> None:
        """To be called by the Nodes after an auction if a load was attributed to the Carriers"""
        self._in_transit = True
        current_node = self._next_node
        current_node.remove_carrier_from_waiting_list(self)
        self._next_node = next_node
        self._time_to_go = self._environment.get_distance(current_node, self._next_node)
        self._load = load  # note that the get_attribution of the load is called by the auction of the node

    def receive_payment(self, value: float) -> None:
        """To be called by the Shippers after an auction if a load was attributed"""
        self._this_episode_revenues += value
        self._total_revenues += value

    def dont_get_attribution(self) -> None:
        """To be called by the Nodes after an auction if the Carriers lost"""
        new_next_node = self._decide_next_node()
        if new_next_node != self._next_node:
            self._in_transit = True
            current_node = self._next_node
            self._next_node = new_next_node
            self._time_to_go = self._environment.get_distance(current_node, self._next_node)
            current_node.remove_carrier_from_waiting_list(self)

    def next_step(self) -> None:
        """To be called by the environment at each iteration"""
        if self._in_transit:
            self._time_to_go -= 1
            new_cost = self._transit_costs() + self._far_from_home_costs()
            if self._time_to_go == 0:
                self._arrive_at_next_node()  # this does not reinitialize the costs trackers nor generate next state
        else:
            new_cost = self._far_from_home_costs()

        self._this_episode_expenses.append(new_cost)
        self._total_expenses += new_cost
        self._update_ffh_cost_functions()

        if not self._in_transit:  # May have been modified by the _arrive_at_next_node method
            self._episode_revenues.append(self._this_episode_revenues)
            self._episode_expenses.append(sum(self._this_episode_expenses))
            self._this_episode_revenues = 0
            self._this_episode_expenses.clear()
            # And generate episode if needed in the LearningCarrier Method

    def _arrive_at_next_node(self) -> None:
        """Called by next_step to do all the variable settings when arrive at a next Nodes
        Note: cost calculations and episode generation are not made here"""
        self._in_transit = False
        if self._load:  # is not none
            self._load.arrive_at_next_node()
        self._load = None
        self._next_node.add_carrier_to_waiting_list(self)

    def clear_load(self) -> None:
        """Called by the environment"""
        self._load = None

    def clear_profit(self) -> None:
        """Called by the environment"""
        self._total_expenses -= sum(self._episode_expenses)  # the expenses of the current episode are still here
        # we will have to delete them at extraction time
        self._episode_expenses.clear()
        self._total_revenues -= sum(self._episode_revenues)  # same
        self._episode_revenues.clear()

    @abc.abstractmethod
    def _transit_costs(self) -> float:
        """Calculating the transit costs depending on the states"""

    @abc.abstractmethod
    def _far_from_home_costs(self) -> float:
        """Calculating the "far from home" costs depending on the states"""

    @abc.abstractmethod
    def _update_ffh_cost_functions(self) -> None:
        """To update your far_from_home costs"""

    @property
    def episode_revenues(self) -> List[float]:
        return self._episode_revenues

    @property
    def episode_expenses(self) -> List[float]:
        return self._episode_expenses

    @property
    def home(self):
        return self._home

    @property
    def name(self):
        return self._name


class CarrierWithCosts(Carrier, abc.ABC):
    """The idea is to modify the Carrier class to have a single cost structure"""

    _cost_dimension: int = 3

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
                 time_not_at_home: int) -> None:
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
        self._time_not_at_home: int = time_not_at_home

    def _transit_costs(self) -> float:
        """The transit costs"""
        return self._t_c

    def _far_from_home_costs(self, time_not_at_home=None) -> float:
        """
        The far from home costs,m calculated with the given value if given, else, calculated with the current
        context
        """
        t = time_not_at_home if time_not_at_home is not None else self._time_not_at_home
        return self._ffh_c + 53.*(exp(self._environment.nb_hours_per_time_unit*0.0015*t) - 1.)\
            if t > 0 else 0.

    def _update_ffh_cost_functions(self) -> None:
        """Here we update the cost parameter AFTER calculating the costs of the current time step"""
        if not self._in_transit and self._next_node == self._home:
            self._time_not_at_home = 0
        else:
            self._time_not_at_home += 1

    @classmethod
    def cost_dimension(cls) -> int:
        return cls._cost_dimension

    @property
    def next_node(self):
        return self._next_node

    @property
    def t_c(self):
        return self._t_c

    @property
    def ffh_c(self):
        return self._ffh_c
