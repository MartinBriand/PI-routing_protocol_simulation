"""
The most basic node you could think of
"""

from PI_RPS.Mechanics.Actors.nodes.node import Node, NodeWeights
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Tools import Info
    from PI_RPS.Mechanics.Environment import Environment


class DummyNode(Node):  # Actually this is not so dummy and will perhaps not change in the future

    def __init__(self,
                 name: str,
                 weights: NodeWeights,
                 nb_info: int,
                 revenues: List[float],
                 environment: 'Environment'):
        super().__init__(name, weights, revenues, environment)

        self._nb_infos: int = nb_info

    def initialize_weights(self) -> None:
        """create structure and initialize the weights and the number of visits to 0. Should be called by the game"""
        self._weights = {}
        for node_i in self._environment.nodes:
            if node_i != self:
                self._weights[node_i] = {}
                for node_j in self._environment.nodes:
                    if node_j != self:
                        self._weights[node_i][node_j] = 0.  # exponential smoothing starting at 0.

    def update_weights_with_new_infos(self, new_infos: List['Info']) -> None:
        """
        This is the method where the nodes has some intelligence
        should no ho to consume info where arrival and start are the same
        """
        for info in new_infos:
            if info.start == info.arrival or info.start == self or info.arrival == self:
                continue  # this is to avoid useless info to be taken
            else:
                w = self._weights[info.arrival][info.start]
                w += (info.cost - w) / self._nb_infos  # we have an exponential smoothing of self.nb_infos
                self._weights[info.arrival][info.start] = w

        for arrival in self._weights:  # decreasing with time
            for intermediate in self._weights[arrival]:
                self._weights[arrival][intermediate] *= ((self._nb_infos - 1) / self._nb_infos)

    def auction_cost(self) -> float:
        """To calculate the auction cost on a demand of the auction, before asking the shipper to pay"""
        return 0.  # yes this is not 0 but still not much
