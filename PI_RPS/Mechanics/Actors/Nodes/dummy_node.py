"""
The most basic node you could think of
"""

from PI_RPS.Mechanics.Actors.Nodes.node import Node, NodeWeights
from typing import TYPE_CHECKING, List, Dict

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Tools.load import Info
    from PI_RPS.Mechanics.Environment.environment import Environment


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
        """create structure and initialize the weights and the number of visits to distance*2000.
        Should be called by the initializer AFTER registering the distance matrix"""
        self._weights = {}
        for arrival in self._environment.nodes:
            if arrival != self:
                self._weights[arrival] = {}
                for departure in self._environment.nodes:
                    if departure != self:  # exponential smoothing starting at 2000*distance
                        if departure != arrival:
                            self._weights[arrival][departure] = self._environment.get_distance(departure=departure,
                                                                                               arrival=arrival)*2000.
                        else:
                            self._weights[arrival][departure] = 0.

    def update_weights_with_new_infos(self, new_infos: List['Info']) -> None:
        """
        This is the method where the Nodes has some intelligence
        should no ho to consume info where arrival and start are the same
        """
        info_arrival_start_dict = {}  # not to reduce the other
        for info in new_infos:
            if info.start == info.arrival or info.start == self or info.arrival == self:
                continue  # this is to avoid useless info to be taken
            else:
                if info.arrival not in info_arrival_start_dict.keys():
                    info_arrival_start_dict[info.arrival] = []
                if info.start not in info_arrival_start_dict[info.arrival]:
                    info_arrival_start_dict[info.arrival].append(info.start)
                w = self._weights[info.arrival][info.start]
                w += (info.cost - w) / self._nb_infos  # we have an exponential smoothing of self.nb_infos
                self._weights[info.arrival][info.start] = w

        for arrival in info_arrival_start_dict:  # decreasing with time
            for intermediate in self._weights[arrival]:
                if intermediate not in info_arrival_start_dict[arrival]:
                    self._weights[arrival][intermediate] *= ((self._nb_infos - 1) / self._nb_infos)

    def auction_cost(self) -> float:
        """To calculate the auction cost on a demand of the auction, before asking the shipper to pay"""
        return 0.  # yes this is not 0 but still not much

    def readable_weights(self) -> Dict:
        result = {}
        for key1 in self._weights.keys():
            result[key1.name] = {}
            for key2 in self._weights[key1]:
                result[key1.name][key2.name] = self._weights[key1][key2]
        return result
