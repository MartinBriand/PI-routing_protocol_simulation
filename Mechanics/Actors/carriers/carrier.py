"""
Carrier file
"""
import abc


class Carrier(abc.ABC):
    """
    A carriers has two states:
        * it is either in transit, if so, it is forced to finish it journey
        * If not, it is because it is at a nodes, where, it can participate in an auction, and depending on the result
            * will either be attributed a good to transport and will have to carry it to the next nodes
            * Or will not get a good and may decide to stay or move to another nodes
    """

    def __init__(self, name, home, in_transit, next_node, time_to_go, load, environment, expenses, revenues):
        self._name = name
        self._home = home
        # state: if not in transit, we are at nodes next_node, ef not time_to_go > 0 and we are going to next_node
        self._in_transit = in_transit
        self._next_node = next_node  # instantiated after the creation of the node
        self._time_to_go = time_to_go
        self._load = load
        self._environment = environment  # instantiated after the creation of the environment

        # costs are allowed to be methods

        self._expenses = expenses
        self._revenues = revenues
        self._total_expenses = sum(self._expenses)
        self._total_revenues = sum(self._revenues)

        self._environment.add_carrier(self)

        if not self._in_transit:  # should be instantiated after the creations of the nodes
            self._next_node.add_carrier_to_waiting_list(self)

    @abc.abstractmethod
    def bid(self, node):  # this should be a dictionary: key is next_node, value is float
        """To be called by the nodes before an auction"""

    def get_attribution(self, load, next_node):
        """To be called by the nodes after an auction if a load was attributed to the carriers"""
        self._in_transit = True
        current_node = self._next_node
        current_node.remove_carrier_from_waiting_list(self)
        self._next_node = next_node

        self._time_to_go = self._environment.get_distance(current_node, self._next_node)
        self._load = load  # note that the get_attribution of the load is called by the auction of the node

    def receive_payment(self, value):
        """To be called by the shippers after an auction if a load was attributed"""
        self._revenues.append(value)
        self._total_revenues += value

    def dont_get_attribution(self):
        """To be called by the nodes after an auction if the carriers lost"""
        new_next_node = self._decide_next_node()
        if new_next_node != self._next_node:
            self._in_transit = True
            current_node = self._next_node
            self._next_node = new_next_node
            self._time_to_go = self._environment.get_distance(current_node, self._next_node)
            current_node.remove_carrier_from_waiting_list(self)

    @abc.abstractmethod
    def _decide_next_node(self):
        """Decide of a next nodes after losing an auction (can be the same nodes when needed)"""

    def next_step(self):
        """To be called by the environment at each iteration"""
        if self._in_transit:
            self._time_to_go -= 1
            new_cost = self._transit_costs() + self._far_from_home_costs()
            if self._time_to_go == 0:
                self._arrive_at_next_node()
        else:
            new_cost = self._far_from_home_costs()

        self._expenses.append(new_cost)
        self._total_expenses += new_cost
        self._update_ffh_cost_functions()

    def _arrive_at_next_node(self):
        """Called by next_step to do all the variable settings when arrive at a next nodes"""
        self._in_transit = False
        if self._load:  # is not none
            self._load.arrive_at_next_node()
        self._load = None
        self._next_node.add_carrier_to_waiting_list(self)

    @abc.abstractmethod
    def _transit_costs(self):
        """Calculating the transit costs depending on the states"""

    @abc.abstractmethod
    def _far_from_home_costs(self):
        """Calculating the "far from home" costs depending on the states"""

    @abc.abstractmethod
    def _update_ffh_cost_functions(self):
        """To update your far_from_home costs"""
