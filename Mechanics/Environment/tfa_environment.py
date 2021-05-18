"""
The tfa environment class
Not sure we will keep it...
"""

from tensorflow.python.framework.ops import EagerTensor
from tensorflow import constant
from numpy import zeros

from Mechanics.Environment.environment import Environment

from typing import TYPE_CHECKING
from prj_typing.types import NodeStates

if TYPE_CHECKING:
    from Mechanics.Actors.nodes.node import Node


class TFAEnvironment(Environment):  # , TFEnvironment):
    """
    Environment with some attributes and builders for the learning context
    """

    def __init__(self, nb_hours_per_time_unit: float) -> None:
        super().__init__(nb_hours_per_time_unit)
        self._node_states: NodeStates = {}

    def this_node_state(self, node: 'Node') -> EagerTensor:
        return self._node_states[node]

    def build_node_state(self) -> None:  # first node in list, then tensor[1, 0, 0, ...], second: [0, 1, 0, ...]
        var = zeros(len(self._nodes), dtype='float32')
        for k in range(len(self._nodes)):
            var[k-1] = 0
            var[k] = 1
            self._node_states[self._nodes[k]] = constant(var)
