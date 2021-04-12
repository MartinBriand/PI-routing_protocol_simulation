"""
The most basic node you could think of
"""
from Mechanics.Actors.nodes.node import Node


class DummyNode(Node):  # Actually this is not so dummy and will perhaps not change in the future

    def __init__(self, name, weights, nb_info, revenues, environment):
        super().__init__(name, weights, revenues, environment)

        self.nb_infos = nb_info

    def initialize_weights(self):
        """create structure and initialize the weights and the number of visits to 0. Should be called by the game"""
        self.nb_infos = {}
        self.weights = {}
        for node_i in self.environment.nodes:
            if node_i != self:
                self.nb_infos[node_i] = {}
                self.weights[node_i] = {}
                for node_j in self.environment.nodes:
                    if node_j != self:
                        self.nb_infos[node_i][node_j] = 0
                        self.weights[node_i][node_j] = 0.

    def update_weights_with_new_infos(self, new_infos):
        """
        This is the method where the nodes has some intelligence
        should no ho to consume info where arrival and start are the same
        """
        for info in new_infos:
            if info.start == info.arrival or info.start == self or info.arrival == self:
                continue
            else:
                w = self.weights[info.arrival][info.start]
                nb = self.nb_infos[info.arrival][info.start]
                w = w*nb+info.cost
                nb += 1
                w /= nb
                self.weights[info.arrival][info.start] = w
                self.nb_infos[info.arrival][info.start] = nb

    def auction_cost(self):
        """To calculate the auction cost on a demand of the auction, before asking the shipper to pay"""
        return 5  # yes this is not 0 but still not much
