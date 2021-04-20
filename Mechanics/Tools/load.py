"""
Load file
"""


class Load:
    """
    A load tracks what is happening to it. It has a shippers, a current carriers, and a route with its associated costs
    """

    def __init__(self, start, arrival, shipper, environment):
        self._start = start
        self._arrival = arrival
        self._shipper = shipper
        self._environment = environment

        self._current_carrier = None
        self._next_node = self._start  # note that if you are not in transit you are at a nodes, and your next_node is
        # also your current node then

        self._in_transit = False
        self._is_arrived = False
        self._is_discarded = False
        self._has_new_infos = False

        self._route_costs = []  # a cost is a tuple (previous_node, next_node, carrier_cost, previous_node_cost)
        self._previous_infos = [Info(self._start, self._start, 0)]  # to calculate the new info, use the old info and
        # add the cost of the current step

        # And now add it to the start nodes and the environment
        self._shipper.add_load(self)
        self._start.add_load_to_waiting_list(self)
        self._environment.add_load(self)

    def get_attribution(self, carrier, previous_node, next_node, carrier_cost, previous_node_cost):
        """
        To be called by the nodes each time a load which was waiting at a nodes get attributed for a next hop
        """
        self._in_transit = True
        self._current_carrier = carrier
        self._next_node.remove_load_from_waiting_list(self)
        self._next_node = next_node
        self._route_costs.append((previous_node, next_node, carrier_cost, previous_node_cost))

        self._new_node_infos(next_node, carrier_cost + previous_node_cost)  # we call the new infos function at each
        # attribution

    def _new_node_infos(self, next_node, cost):
        """
        Generate new info after the attribution and tell the environment it has new info
        """
        infos = []
        for info in self._previous_infos:
            infos.append(Info(info.start, next_node, info.cost + cost))
            # tolerance for not writing getters on the info class
            # Even if you go back to yourself, it is important to have the info

        infos.append(Info(next_node, next_node, 0))

        self._previous_infos = infos
        self._has_new_infos = True
        self._environment.add_load_to_new_infos_list(self)

    def communicate_infos(self):
        """Communicate the new info to the environment when asked to"""
        self._has_new_infos = False
        return self._previous_infos

    def arrive_at_next_node(self):
        """
        to be called by the carriers each time it arrives at a next nodes
        """
        self._current_carrier = None
        self._in_transit = False
        if self._next_node == self._arrival:
            self._is_arrived = True
        else:
            self._next_node.add_load_to_waiting_list(self)

    def discard(self):
        """
        Set the load as discarded. Called by the nodes when auction run but no result
        (A shippers could eventually also call it but it is not implemented yet (and will probably not be))
        """
        self._is_discarded = True
        self._next_node.remove_load_from_waiting_list()

    def has_new_infos(self):
        """access the has new infos variable. Called by the environment"""
        return self._has_new_infos

    @property
    def arrival(self):
        return self._arrival

    @property
    def shipper(self):
        return self._shipper


class Info:
    """
    An info is made of a start position, an arrival position, and a cost between the two
    """

    def __init__(self, start, arrival, cost):
        self._start = start
        self._arrival = arrival
        self._cost = cost

    @property
    def start(self):
        return self._start

    @property
    def arrival(self):
        return self._arrival

    @property
    def cost(self):
        return self._cost
