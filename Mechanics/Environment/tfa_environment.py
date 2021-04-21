"""
The tfa environment class
Not sure we will keep it...
"""


from Mechanics.Environment.environment import Environment
from tensorflow import Tensor

from typing import TYPE_CHECKING
from prj_typing.types import NodeState

if TYPE_CHECKING:
    from Mechanics.Actors.nodes.node import Node


class TFAEnvironment(Environment):  # , TFEnvironment):
    """
    Environment with some attributes and builders for the learning context
    """

    def __init__(self) -> None:
        super().__init__()
        self._node_state: NodeState = {}

    def this_node_state(self, node: 'Node') -> Tensor:
        return self._node_state[node]
