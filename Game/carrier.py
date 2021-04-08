"""
Carrier file
"""


class Carrier:
    """
    A carrier has two states:
        * it is either in transit, if so, it is forced to finish it journey
        * If not, it is because it is at a node, where, it can participate in an auction, and depending on the result
            * will either be attributed a good to transport and will have to carry it to the next node
            * Or will not get a good and may decide to stay or move to another node
    """

    def __init__(self, in_transit, next_node, time_to_go, load, environment, expenses, revenues):
        # state: if not in transit, we are at node next_node, ef not time_to_go > 0 and we are going to next_node
        self.in_transit = in_transit
        self.next_node = next_node
        self.time_to_go = time_to_go
        self.load = load
        self.environment = environment

        # costs are allowed to be methods

        self.expenses = expenses
        self.revenues = revenues
        self.total_expenses = sum(self.expenses)
        self.total_revenues = sum(self.revenues)

        if not self.in_transit:
            self.next_node.add_carrier_to_waiting_list(self)

    def bid(self):
        """To be called by the node before an auction"""
        raise NotImplementedError

    def get_attribution(self, load, next_node):
        """To be called by the node after an auction if a load was attributed to the carrier"""
        self.in_transit = True
        current_node = self.next_node
        self.next_node = next_node

        self.time_to_go = self.environment.get_distance(current_node, self.next_node)
        self.load = load  # note that the get_attribution of the load is called by the node too.

    def receive_payment(self, value):
        """To be called by the shipper after an auction if a load was attributed"""
        self.revenues.append(value)
        self.total_revenues += value

    def dont_get_attribution(self):
        """To be called by the node after an auction if the carrier lost"""
        new_next_node = self._decide_next_node()
        if new_next_node != self.next_node:
            self.in_transit = True
            current_node = self.next_node
            self.next_node = new_next_node
            self.time_to_go = self.environment.get_distance(current_node, self.next_node)
            current_node.remove_carrier_from_waiting_list()

    def _decide_next_node(self):
        """Decide of a next node after losing an auction (can be the same node when needed)"""
        raise NotImplementedError

    def next_step(self):
        """To be called by the environment at each iteration"""
        if self.in_transit:
            self.time_to_go -= 1
            new_cost = self._transit_costs() + self._far_from_home_costs()
            if self.time_to_go == 0:
                self._arrive_at_next_node()
        else:
            new_cost = self._far_from_home_costs()

        self.expenses.append(new_cost)
        self.total_expenses += new_cost

    def _arrive_at_next_node(self):
        """Called by next_step to do all the variable settings when arrive at a next node"""
        self.in_transit = False
        self.load.arrive_at_next_node()
        self.load = None
        self.next_node.add_carrier_to_waiting_list(self)  # TODO: Is this implemented in the node API

    def _transit_costs(self):
        raise NotImplementedError

    def _far_from_home_costs(self):
        raise NotImplementedError
