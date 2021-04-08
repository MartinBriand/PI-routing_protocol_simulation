"""
Load file
"""


class Load:
    """
    A load tracks what is happening to it. It has a shipper, a current carrier, and a route with its associated costs
    """

    def __init__(self, start, arrival, shipper, environment):
        self.start = start
        self.arrival = arrival
        self.shipper = shipper
        self.environment = environment

        self.current_carrier = None
        self.next_node = self.start  # note that if you are not in transit, then you are at a node, and you next_node is
        # also your current node

        self.in_transit = False
        self.is_arrived = False
        self.is_discarded = False
        self.has_new_infos = False

        self.route_costs = []  # a cost is a tuple (previous_node, next_node, carrier_cost, previous_node_cost)
        self.previous_info = [Info(self.start, self.start, 0)]  # to calculate the new info, use the old info and add
        # the cost of the current step

        # And now add it to the start node
        self.start.add_load_to_waiting_list(self)

    def get_attribution(self, carrier, previous_node, next_node, carrier_cost, previous_node_cost):
        """
        To be called each time a load which was waiting at a node get attributed for a next hop
        """
        self.in_transit = True
        self.current_carrier = carrier
        self.next_node = next_node
        self.route_costs.append((previous_node, next_node, carrier_cost, previous_node_cost))

        self._new_node_infos(next_node, carrier_cost, previous_node_cost)  # we call the new info function at each
        # attribution

        # TODO: Is this implemented in the node API

    def _new_node_infos(self, next_node, carrier_cost, previous_node_cost):
        """
        Generate new info after the attribution and tell the environment it has new info
        """
        infos = []
        for info in self.previous_info:
            infos.append(Info(info.start, next_node, info.cost + carrier_cost + previous_node_cost))
            # tolerence for not writing getters on the info class

        infos.append(Info(next_node, next_node, 0))

        self.previous_info = infos
        self.has_new_info = True
        self.environment.add_load_to_new_info_list(self)

    def communicate_infos(self):
        """Communicate the new info to the environment when asked to"""
        if not self.has_new_info:
            raise BrokenPipeError('This load has no new info to communicate')
        else:
            self.has_new_infos = False
            return self.previous_info

    def arrive_at_next_node(self):  # TODO: Is this implemented in the carrier
        """
        to be called by the carrier each time it arrives at a next node
        """
        self.current_carrier = None
        self.in_transit = False
        if self.next_node == self.arrival:
            self.is_arrived = True
        else:
            self.next_node.add_load_to_waiting_list(self)

    def discard(self):
        """Set the load as discarded"""
        self.is_discarded = True

    def has_new_infos(self):
        return self.has_new_infos


class Info:
    """
    An info is made of a start position, an arrival position, and a cost between the two
    """

    def __init__(self, start, arrival, cost):
        self.start = start
        self.arrival = arrival
        self.cost = cost
