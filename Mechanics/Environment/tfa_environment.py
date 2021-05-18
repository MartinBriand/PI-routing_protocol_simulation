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

    def __init__(self,
                 nb_hours_per_time_unit: float,
                 t_c_mu: float,
                 t_c_sigma: float,
                 ffh_c_mu: float,
                 ffh_c_sigma: float,
                 tnah_divisor: int
                 ) -> None:
        super().__init__(nb_hours_per_time_unit)
        self._t_c_mu: float = t_c_mu * self._nb_hours_per_time_unit
        self._t_c_sigma: float = t_c_sigma * self._nb_hours_per_time_unit
        self._ffh_c_mu: float = ffh_c_mu * self._nb_hours_per_time_unit
        self._ffh_c_sigma: float = ffh_c_sigma * self._nb_hours_per_time_unit
        self._tnah_divisor: int = tnah_divisor
        self._node_states: NodeStates = {}

    def this_node_state(self, node: 'Node') -> EagerTensor:
        return self._node_states[node]

    def build_node_state(self) -> None:  # first node in list, then tensor[1, 0, 0, ...], second: [0, 1, 0, ...]
        var = zeros(len(self._nodes), dtype='float32')
        for k in range(len(self._nodes)):
            var[k - 1] = 0
            var[k] = 1
            self._node_states[self._nodes[k]] = constant(var)

    @property
    def t_c_mu(self) -> float:
        return self._t_c_mu

    @property
    def t_c_sigma(self) -> float:
        return self._t_c_sigma

    @property
    def ffh_c_mu(self) -> float:
        return self._ffh_c_mu

    @property
    def ffh_c_sigma(self) -> float:
        return self._ffh_c_sigma

    @property
    def tnah_divisor(self) -> int:
        return self._tnah_divisor
