"""
Carrier file
"""
import abc
from typing import TYPE_CHECKING, Optional, List, Tuple
import random

from PI_RPS.prj_typing.types import CarrierBid, CarrierMultiBid, CarrierSingleBid

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Tools.load import Load
    from PI_RPS.Mechanics.Environment.environment import Environment


class Carrier(abc.ABC):
    """
    A Carriers has two states:
        * it is either in transit, if so, it is forced to finish its journey
        * If not, it is because it is at a Node, where, it can participate in an auction, and depending on the result
            * will either be attributed a good to transport and will have to carry it to the next Nodes
            * Or will not get nothing and may decide to stay at the present NOde or move to another one
    Note that far_from_home_costs are what we call NonRoadCosts in the paper.
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
                 this_episode_revenues: float) -> None:

        self._name: str = name
        self._home: 'Node' = home
        # state: if not in transit, we are at Nodes next_node, ef not time_to_go > 0 and we are going to next_node
        self._in_transit: bool = in_transit
        self._previous_node: 'Node' = previous_node
        self._next_node: 'Node' = next_node  # instantiated after the creation of the node
        self._time_to_go: int = time_to_go
        self._load: 'Load' = load
        self._environment: 'Environment' = environment  # instantiated after the creation of the environment

        # costs are allowed to be methods

        self._episode_types: List[Tuple[str, 'Node', 'Node', bool]] = episode_types
        # Transport, Wait, Empty  - origin  - dest
        self._episode_expenses: List[float] = episode_expenses
        self._episode_revenues: List[float] = episode_revenues
        self._this_episode_expenses: List[float] = this_episode_expenses
        self._this_episode_revenues: float = this_episode_revenues
        self._total_expenses: float = sum(self._episode_expenses) + sum(self._this_episode_expenses)
        self._total_revenues: float = sum(self._episode_revenues) + self._this_episode_revenues

        self._environment.add_carrier(self)

        if not self._in_transit:  # should be instantiated after the creations of the Nodes
            self._next_node.add_carrier_to_waiting_list(self)

        self._reserve_price_involved_in_transition = False

    @abc.abstractmethod
    def bid(self, *args, **kwargs) -> CarrierBid:
        """To be called by the Auctions run by node"""
        raise NotImplementedError

    @abc.abstractmethod
    def _decide_next_node(self) -> 'Node':
        """Decide of a next Node after losing an auction (can be the same Node when needed)"""
        raise NotImplementedError

    def get_attribution(self, load: 'Load', next_node: 'Node', reserve_price_involved: bool) -> None:
        """To be called by the Auction after running if a load was attributed to this Carrier"""
        self._in_transit = True
        self._previous_node.remove_carrier_from_waiting_list(self)
        self._next_node = next_node
        self._time_to_go = self._environment.get_distance(self._previous_node, self._next_node)
        self._load = load  # note that the get_attribution of the load is called by the auction of the node
        self._reserve_price_involved_in_transition = reserve_price_involved

    def receive_payment(self, value: float) -> None:
        """To be called by the Shipper of the attributed load after a won auction"""
        self._this_episode_revenues += value
        self._total_revenues += value

    def dont_get_attribution(self) -> None:
        """To be called by the Auction after running if this Carrier lost"""
        new_next_node = self._decide_next_node()
        if new_next_node != self._next_node:
            self._in_transit = True
            self._next_node = new_next_node
            self._time_to_go = self._environment.get_distance(self._previous_node, self._next_node)
            self._previous_node.remove_carrier_from_waiting_list(self)

    def next_step(self) -> None:
        """To be called by the environment at each iteration to update the carrier state"""
        if self._in_transit:
            self._time_to_go -= 1
            new_cost = self._transit_costs() + self._far_from_home_costs()
            if self._time_to_go == 0:
                self._arrive_at_next_node()  # this does not reinitialize the costs trackers nor generate next state
        else:
            new_cost = self._far_from_home_costs()
            self._episode_types.append(('Wait', self._previous_node, self._next_node, False))

        self._this_episode_expenses.append(new_cost)
        self._total_expenses += new_cost

        if not self._in_transit:  # May have been modified by the _arrive_at_next_node method
            self._episode_revenues.append(self._this_episode_revenues)
            self._episode_expenses.append(sum(self._this_episode_expenses))
            self._this_episode_revenues = 0.
            self._this_episode_expenses.clear()

    def _arrive_at_next_node(self) -> None:
        """Called by next_step to do all the variable settings when arriving at a next Node"""
        self._in_transit = False
        if self._load:  # is not none
            self._load.arrive_at_next_node()
            self._episode_types.append(('Transport',
                                        self._previous_node,
                                        self._next_node,
                                        self._reserve_price_involved_in_transition))
        else:
            self._episode_types.append(('Empty', self._previous_node, self._next_node, False))
        self._load = None
        self._next_node.add_carrier_to_waiting_list(self)
        self._previous_node = self._next_node

    def clear_load(self) -> None:
        """Called by the environment to clear the load (memory saver)"""
        self._load = None

    def clear_profit(self) -> None:
        """Called by the environment to clear profit for new series"""
        self._total_expenses -= sum(self._episode_expenses)  # the expenses of the current episode are still here
        # we will have to delete them at extraction time
        self._episode_expenses.clear()
        self._total_revenues -= sum(self._episode_revenues)  # same
        self._episode_revenues.clear()
        self._episode_types.clear()

    @abc.abstractmethod
    def _transit_costs(self) -> float:
        """Calculating the transit costs depending on the context of the Carrier"""
        raise NotImplementedError

    @abc.abstractmethod
    def _far_from_home_costs(self) -> float:
        """Calculating the "far from home" costs depending on the context of the Carrier"""
        raise NotImplementedError

    @property
    def episode_types(self) -> List[Tuple[str, 'Node', 'Node', bool]]:
        return self._episode_types

    @property
    def episode_revenues(self) -> List[float]:
        return self._episode_revenues

    @property
    def episode_expenses(self) -> List[float]:
        return self._episode_expenses

    @property
    def home(self) -> 'Node':
        return self._home

    @property
    def name(self) -> str:
        return self._name


# Bidding abstract classes

class MultiBidCarrier(Carrier, abc.ABC):
    """
    A carrier bidding on multiple lanes (abstract method)
    """
    @abc.abstractmethod
    def bid(self) -> CarrierMultiBid:
        raise NotImplementedError


class SingleBidCarrier(Carrier, abc.ABC):
    """
    A carrier bidding on a single lane. If they all bid on the destination lane, then it forms the control group
    """
    @abc.abstractmethod
    def bid(self, next_node: 'Node') -> CarrierSingleBid:
        raise NotImplementedError


# Cost abstract classes

class CarrierWithCosts(Carrier, abc.ABC):
    """
    Carrier with the cost structure of the model
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
                 transit_cost: Optional[float] = None,
                 far_from_home_cost: Optional[float] = None,
                 time_not_at_home: int = 0) -> None:
        super().__init__(name=name,
                         home=home,
                         in_transit=in_transit,
                         previous_node=previous_node,
                         next_node=next_node,
                         time_to_go=time_to_go,
                         load=load,
                         environment=environment,
                         episode_types=episode_types,
                         episode_expenses=episode_expenses,
                         episode_revenues=episode_revenues,
                         this_episode_expenses=this_episode_expenses,
                         this_episode_revenues=this_episode_revenues)

        if transit_cost is None and far_from_home_cost is None:
            self.random_new_cost_parameters()
        elif transit_cost is not None and far_from_home_cost is not None:
            self._t_c: float = transit_cost
            self._ffh_c: float = far_from_home_cost
        else:
            raise ValueError("transit_cost and far_from_home_costs should either both be None or both have value")
        self._a_c_percentage: float = 0.2

        self._time_not_at_home: int = time_not_at_home

    def next_step(self) -> None:
        """Same but updating the cost function to increment or not the time not at home"""
        super().next_step()
        self._update_ffh_cost_functions()

    def _transit_costs(self) -> float:
        """The transit costs"""
        return self._t_c

    def _far_from_home_costs(self, time_not_at_home=None) -> float:
        """
        The far from home costs (or the non road costs in the paper)
        """
        t = time_not_at_home if time_not_at_home is not None else self._time_not_at_home
        return self._ffh_c + 53. + self._a_c_percentage * (self._ffh_c + self._t_c) if t > 0 else 0.
        # why do we have this +53. ?????

    def _update_ffh_cost_functions(self) -> None:
        """
        To update the cost function (actually in our case it doesn't update anything, but it could).
        """
        if not self._in_transit and self._next_node == self._home:
            self._time_not_at_home = 0
        else:
            self._time_not_at_home += 1

    def random_new_cost_parameters(self) -> None:
        """Selecting new cost parameters and setting them"""
        road_costs = random.normalvariate(mu=self._environment.t_c_mu, sigma=self._environment.t_c_sigma)
        drivers_costs = random.normalvariate(mu=self._environment.ffh_c_mu, sigma=self._environment.ffh_c_sigma)
        self._set_new_cost_parameters(t_c=road_costs, ffh_c=drivers_costs)

    def _set_new_cost_parameters(self, t_c: float, ffh_c: float) -> None:
        """
        Setting new parameters for learners and resetting profit lists
        """
        self._t_c = t_c
        self._ffh_c = ffh_c
        self.clear_profit()

    @property
    def next_node(self) -> 'Node':
        return self._next_node

    @property
    def t_c(self) -> float:
        return self._t_c

    @property
    def ffh_c(self) -> float:
        return self._ffh_c
