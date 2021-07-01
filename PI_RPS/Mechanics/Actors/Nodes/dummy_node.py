"""
The most basic node you could think of
"""

from PI_RPS.Mechanics.Actors.Nodes.node import Node, NodeWeights
from typing import TYPE_CHECKING, List, Dict, Optional

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Tools.load import Info
    from PI_RPS.Mechanics.Environment.environment import Environment


class DummyNode(Node):  # Actually this is not so dummy and will perhaps not change in the future

    def __init__(self,
                 name: str,
                 weight_master: 'DummyNodeWeightMaster',
                 revenues: List[float],
                 environment: 'Environment',
                 auction_cost: float,
                 auction_type: str,
                 ):
        super().__init__(name, {}, revenues, environment, auction_type)

        self._auction_cost: float = auction_cost
        self._weight_master: 'DummyNodeWeightMaster' = weight_master
        self._weight_master.register_node(self)

    def update_weights_with_new_infos(self, new_infos: List['Info']) -> None:
        self._weight_master.update_weights_with_new_infos(self, new_infos)

    def auction_cost(self) -> float:
        """To calculate the auction cost on a demand of the auction, before asking the shipper to pay"""
        return self._auction_cost

    def set_weights(self, departure: 'DummyNode', arrival: 'DummyNode', value: float) -> None:
        """Called by the weight master to change the weights"""
        # assert departure != self  # should not be called, remove for speed reasons
        self._weights[arrival][departure] = value

    def set_init_weights(self, departure: 'DummyNode', arrival: 'DummyNode', value: float) -> None:
        """Called by the weight master at weight initialization"""
        assert arrival != self and departure != self, "Cannot create self weight"
        if arrival not in self._weights:
            self._weights[arrival] = {}
        assert departure not in self._weights[arrival].keys(), "Connexion already created"
        self._weights[arrival][departure] = value

    def delete_weights(self):
        """Called during some training"""
        self._weights = {}

    @property
    def weight_master(self) -> 'DummyNodeWeightMaster':
        return self._weight_master

    @property
    def is_learning(self) -> bool:
        return self._weight_master.is_learning


class DummyNodeWeightMaster:
    """To centralize the learning process for efficiency reasons"""

    def __init__(self,
                 environment: 'Environment',
                 nb_infos: int,
                 is_learning: bool,
                 nodes: Optional[List['DummyNode']] = None) -> None:

        self._environment: 'Environment' = environment
        self._nb_infos: int = nb_infos

        self._weights: NodeWeights = {}
        self._is_initialized: bool = False

        self._nodes: List['DummyNode'] = nodes if nodes else []

        self._is_learning: bool = is_learning

        self._has_learned: bool = False

        if nodes:
            self._has_asked_to_learn: Dict['DummyNode', bool] = {node: False for node in self._nodes}
        else:
            self._has_asked_to_learn = {}

    def register_node(self, node: 'DummyNode') -> None:
        """Register a node to the weight master. Called by the DummyNode"""
        assert not self._is_initialized, "Can't add when already initialized"
        assert node not in self._nodes, "Can't add node if already in list of nodes"
        self._nodes.append(node)
        self._has_asked_to_learn[node] = False

    def initialize(self, weights: Optional['NodeWeights'] = None) -> None:
        """create structure and initialize the weights and the number of visits to distance*2000.
        Should be called by the initializer AFTER registering the distance matrix"""
        assert not self._is_initialized, "Can initialize only once"
        assert all(not i for i in self._has_asked_to_learn.values()), "No learning"
        if weights:
            self._weights = weights
        else:
            self._weights = {}
            for arrival in self._environment.nodes:
                self._weights[arrival] = {}
                for departure in self._environment.nodes:
                    if departure != arrival:
                        self._weights[arrival][departure] = \
                            (self._environment.init_node_weights_distance_scaling_factor *
                             self._environment.get_distance(departure=departure,
                                                            arrival=arrival))
                    else:
                        self._weights[arrival][departure] = 0.

        self._broadcast_init_weights()
        self._is_initialized = True

    def reinitialize(self, weights: Optional['NodeWeights'] = None) -> None:
        """Will be called during some training"""
        for node in self._nodes:
            node.delete_weights()
        self._is_initialized = False
        self.initialize(weights=weights)

    def update_weights_with_new_infos(self, node: 'DummyNode', new_infos: List['Info']) -> None:
        """
        This is the method where the Nodes has some intelligence
        should not consume info where arrival and start are the same
        Learn only for all nodes
        Broadcast weights when learning is finished
        """
        if self._is_learning:
            assert not self._has_asked_to_learn[node], "Can't ask to learn twice"
            if not self._has_learned:
                info_start_arrival_dict = {}  # not to reduce the other
                for info in new_infos:
                    if info.start == info.arrival:
                        continue  # this is to avoid useless info to be taken
                    else:
                        if info.cost <= self._environment.get_distance(info.start, info.arrival) * \
                                self._environment.max_node_weights_distance_scaling_factor:
                            if info.start not in info_start_arrival_dict.keys():
                                info_start_arrival_dict[info.start] = {}
                            if info.arrival not in info_start_arrival_dict[info.start].keys():
                                info_start_arrival_dict[info.start][info.arrival] = []
                            info_start_arrival_dict[info.start][info.arrival].append(info.cost)
                for start in info_start_arrival_dict.keys():
                    for arrival in info_start_arrival_dict[start].keys():
                        w = self._weights[arrival][start]
                        infos = info_start_arrival_dict[start][arrival]
                        nb_infos = len(infos)
                        value = sum(infos) / nb_infos

                        w += (value - w) / (
                                self._nb_infos ** (1 / nb_infos))  # we have an exponential smoothing of self.nb_infos
                        self._weights[arrival][start] = w
                self._has_learned = True
            self._has_asked_to_learn[node] = True

            if all(i for i in self._has_asked_to_learn.values()):
                self._broadcast_weights()
                self._has_learned = False
                for node in self._has_asked_to_learn.keys():
                    self._has_asked_to_learn[node] = False

    def _broadcast_weights(self) -> None:
        """
        Broadcast the weights to the DummyNodes
        """
        for node in self._nodes:
            for arrival in self._weights.keys():
                if arrival != node:
                    for departure in self._weights[arrival].keys():
                        if departure != node:
                            node.set_weights(departure, arrival, self._weights[arrival][departure])

    def _broadcast_init_weights(self) -> None:
        """
        Broadcast the weights to the DummyNodes at initialization
        """
        for node in self._nodes:
            for arrival in self._weights.keys():
                if arrival != node:
                    for departure in self._weights[arrival].keys():
                        if departure != node:
                            node.set_init_weights(departure, arrival, self._weights[arrival][departure])

    def readable_weights(self) -> Dict:
        result = {}
        for key1 in self._weights.keys():
            result[key1.name] = {}
            for key2 in self._weights[key1]:
                result[key1.name][key2.name] = self._weights[key1][key2]
        return result

    def weights_text(self):
        return {key1.name: {key2.name: value2 for key2, value2 in value1.items()}
                for key1, value1 in self._weights.items()}

    @property
    def is_learning(self) -> bool:
        return self._is_learning

    @is_learning.setter
    def is_learning(self, value) -> None:
        assert type(value) == bool, 'only set booleans'
        assert self._is_learning == (not value), 'only change to opposite'
        self._is_learning = value

    @property
    def nb_infos(self) -> int:
        return self._nb_infos
