"""
Load file
"""


class Load:
    """
    A load tracks what is happening to it. It has a shipper, a current carrier, and a route with its associated costs
    """

    def __init__(self, start, arrival, shipper):
        self.start = start
        self.arrival = arrival
        self.shipper = shipper
        self.current_carrier = None
        self.next_node = self.start  # note that if you are not in transit, then you are at a node, and you next_node is
        # also your current node
        self.in_transit = False
        self.is_arrived = False

        self.route_costs = []  # a cost is a tuple (previous_node, next_node, carrier_cost, previous_node_cost)
        self.previous_info = [Info(self.start, self.start, 0)]  # to calculate the new info, use the old info and add
        # the cost of the current step

    def get_attribution(self, carrier, previous_node, next_node, carrier_cost, previous_node_cost):
        self.in_transit = True
        self.current_carrier = carrier
        self.route_costs.append((previous_node, next_node, carrier_cost, previous_node_cost))

        self.new_node_infos(next_node, carrier_cost, previous_node_cost)  # we call the new info function at each
        # attribution

    def arrive_at_next_node(self):
        pass  # TODO: implement this function

    def new_node_infos(self, next_node, carrier_cost, previous_node_cost):
        infos = []
        for info in self.previous_info:
            infos.append(Info(info.start, next_node, info.cost + carrier_cost + previous_node_cost))

        infos.append(Info(next_node, next_node, 0))

        self.previous_info = infos
        # TODO: pass the info to all the nodes


class Info:
    """
    An info is made of a start position, an arrival position, and a cost between the two
    """

    def __init__(self, start, arrival, cost):
        self.start = start
        self.arrival = arrival
        self.cost = cost
