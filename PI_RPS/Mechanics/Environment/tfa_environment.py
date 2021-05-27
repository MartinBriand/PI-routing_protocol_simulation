"""
The tfa environment class
Not sure we will keep it...
"""

from tensorflow.python.framework.ops import EagerTensor
from tensorflow import constant
from numpy import zeros
import random

from PI_RPS.Mechanics.Environment.environment import Environment

from typing import TYPE_CHECKING, List
from PI_RPS.prj_typing.types import NodeStates

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Actors.Carriers.learning_carrier import LearningAgent, LearningCarrier


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
                 max_nb_infos_per_load: int,
                 tnah_divisor: float,
                 action_min: float,
                 action_max: float,
                 init_node_weights_distance_scaling_factor: float,
                 max_time_not_at_home: int
                 ) -> None:
        super().__init__(nb_hours_per_time_unit, max_nb_infos_per_load, init_node_weights_distance_scaling_factor)
        self._t_c_mu: float = t_c_mu * self._nb_hours_per_time_unit
        self._t_c_sigma: float = t_c_sigma * self._nb_hours_per_time_unit
        self._ffh_c_mu: float = ffh_c_mu * self._nb_hours_per_time_unit
        self._ffh_c_sigma: float = ffh_c_sigma * self._nb_hours_per_time_unit
        self._tnah_divisor: float = tnah_divisor
        self._action_min: float = action_min
        self._action_max: float = action_max
        self._max_time_not_at_home = max_time_not_at_home
        self._enough_transitions_carriers: List['LearningCarrier'] = []
        self._node_states: NodeStates = {}
        self._learning_agent = None

    def this_node_state(self, node: 'Node') -> EagerTensor:
        """Return the state of the present node"""
        return self._node_states[node]

    def build_node_state(self) -> None:
        """first node in list: [1, 0, 0, ...], second: [0, 1, 0, ...]"""
        var = zeros(len(self._nodes), dtype='float32')
        for k in range(len(self._nodes)):
            var[k - 1] = 0
            var[k] = 1
            self._node_states[self._nodes[k]] = constant(var.copy())

    def register_learning_agent(self, learning_agent: 'LearningAgent') -> None:
        """
        Called by the learning agent to signal its presence to the environment
        """
        self._learning_agent = learning_agent

    def add_carrier_to_enough_transitions(self, carrier: 'LearningCarrier') -> None:
        """
        To be called by the Learning Carrier to update the list
        Do not forget to clean the list after training
        """
        self._enough_transitions_carriers.append(carrier)

    def shuffle_enough_transitions_carriers(self) -> None:
        random.shuffle(self._enough_transitions_carriers)

    def pop_enough_transitions_carriers(self) -> 'LearningCarrier':
        return self._enough_transitions_carriers.pop()

    def check_carriers_first_steps(self) -> None:
        for carrier in self._carriers:
            carrier.init_first_step()

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
    def tnah_divisor(self) -> float:
        return self._tnah_divisor

    @property
    def learning_agent(self) -> 'LearningAgent':
        return self._learning_agent

    @property
    def action_min(self) -> float:
        return self._action_min

    @property
    def action_max(self) -> float:
        return self._action_max

    @property
    def max_time_not_at_home(self) -> int:
        return self._max_time_not_at_home

    @property
    def enough_transitions_carriers(self) -> List['LearningCarrier']:
        return self._enough_transitions_carriers
