"""
Load file
"""


class Load:
    """
    A load tracks what is happening to it. It has a shippers, a current carriers, and a route with its associated costs
    """

    def __init__(self, start, arrival, shipper, environment):
        self.start = start
        self.arrival = arrival
        self.shipper = shipper
        self.environment = environment

        self.current_carrier = None
        self.next_node = self.start  # note that if you are not in transit you are at a nodes, and your next_node is
        # also your current node then

        self.in_transit = False
        self.is_arrived = False
        self.is_discarded = False
        self._has_new_infos = False

        self.route_costs = []  # a cost is a tuple (previous_node, next_node, carrier_cost, previous_node_cost)
        self.previous_infos = [Info(self.start, self.start, 0)]  # to calculate the new info, use the old info and add
        # the cost of the current step

        # And now add it to the start nodes and the environment
        self.shipper.add_load(self)
        self.start.add_load_to_waiting_list(self)
        self.environment.add_load(self)

    def get_attribution(self, carrier, previous_node, next_node, carrier_cost, previous_node_cost):
        """
        To be called by the nodes each time a load which was waiting at a nodes get attributed for a next hop
        """
        self.in_transit = True
        self.current_carrier = carrier
        self.next_node.remove_load_from_waiting_list(self)
        self.next_node = next_node
        self.route_costs.append((previous_node, next_node, carrier_cost, previous_node_cost))

        self._new_node_infos(next_node, carrier_cost + previous_node_cost)  # we call the new infos function at each
        # attribution

    def _new_node_infos(self, next_node, cost):
        """
        Generate new info after the attribution and tell the environment it has new info
        """
        infos = []
        for info in self.previous_infos:
            infos.append(Info(info.start, next_node, info.cost + cost))
            # tolerance for not writing getters on the info class
            # Even if you go back to yourself, it is important to have the info

        infos.append(Info(next_node, next_node, 0))

        self.previous_infos = infos
        self._has_new_infos = True
        self.environment.add_load_to_new_infos_list(self)

    def communicate_infos(self):
        """Communicate the new info to the environment when asked to"""
        self._has_new_infos = False
        return self.previous_infos

    def arrive_at_next_node(self):
        """
        to be called by the carriers each time it arrives at a next nodes
        """
        self.current_carrier = None
        self.in_transit = False
        if self.next_node == self.arrival:
            self.is_arrived = True
        else:
            self.next_node.add_load_to_waiting_list(self)

    def discard(self):
        """
        Set the load as discarded. Called by the nodes when auction run but no result
        (A shippers could eventually also call it but it is not implemented yet (and will probably not be))
        """
        self.is_discarded = True
        self.next_node.remove_load_from_waiting_list()

    def has_new_infos(self):
        """access the has new infos variable. Called by the environment"""
        return self._has_new_infos


class Info:
    """
    An info is made of a start position, an arrival position, and a cost between the two
    """

    def __init__(self, start, arrival, cost):
        self.start = start
        self.arrival = arrival
        self.cost = cost
