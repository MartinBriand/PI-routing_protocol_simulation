"""
This is the carrier bidding its costs but only on the lane of the load's destination
"""


class SingleLaneCostBiddingCarrier():
    """Write description"""

    def __init__(self):
        pass

    def _decide_next_node(self) -> 'Node':
        """
        Go home only if more than self._max_time_not_at_home since last time at home
        """

        if self._time_not_at_home > self._max_time_not_at_home:
            return self._home
        else:
            return self._next_node

    def bid(self, node: 'Node') -> 'CarrierBid':
        bid = {}
        if self._time_not_at_home > self._max_time_not_at_home:
            for next_node in self._environment.nodes:
                if next_node != node:
                    bid[next_node] = 0 if next_node == self._home else self._too_high_bid
        else:
            for next_node in self._environment.nodes:
                if next_node != node:
                    bid[next_node] = self._calculate_costs(node, next_node)

        return bid

    def _calculate_costs(self, from_node: 'Node', to_node: 'Node') -> float:
        result = 0.
        for delta_t in range(self._environment.get_distance(from_node, to_node)):
            t = self._time_not_at_home + delta_t
            result += self._transit_costs() + self._far_from_home_costs(time_not_at_home=t)
        return result
