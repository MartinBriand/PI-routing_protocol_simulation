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

    def __init__(self):
        pass

    def bid(self):
        """To be called by the node before an auction"""
        raise NotImplementedError

    def get_attribution(self):
        """To be called by the node after an auction if a load was attributed to the carrier"""
        pass

    def receive_payment(self):
        """To be called by the shipper after an auction if a load was attributed"""
        pass

    def dont_get_attribution(self):
        """To be called by the node after an auction if the carrier lost"""
        self.next_node = self._decide_next_node()
        if self.next_node != self.current_node:
            pass

    def _decide_next_node(self):
        """Decide of a next node after losing an auction (can be the same node when needed)"""
        raise NotImplementedError

    def next_step(self):
        """To be called by the environment at each iteration"""
        pass

    def _arrive_at_next_node(self):
        """Called by next_step to do all the variable settings when arrive at a next node"""
        pass
