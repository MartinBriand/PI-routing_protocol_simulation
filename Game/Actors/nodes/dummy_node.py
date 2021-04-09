"""
The most basic node you could think of
"""
from Game.Actors.nodes.node import Node


class DummyNode(Node):

    def update_weights_with_new_infos(self, new_infos):  # TODO: implement this method
        """
        This is the method where the nodes has some intelligence
        """
        raise NotImplementedError

    def auction_cost(self):
        """To calculate the auction cost on a demand of the auction, before asking the shipper to pay"""
        return 5  # yes this is not 0 but still not much
