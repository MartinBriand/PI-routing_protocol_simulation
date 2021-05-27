"""
The most basic shipper you could think of
"""

from PI_RPS.Mechanics.Actors.Shippers.shipper import Shipper
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Tools.load import Load


class DummyShipper(Shipper):

    def __init__(self,
                 name: str,
                 laws: List['NodeLaw'],  # forward reference
                 expenses: List[float],
                 loads: List['Load'],
                 environment: 'Environment',
                 reserve_price: float) -> None:

        super().__init__(name,
                         laws,  # forward reference
                         expenses,
                         loads,
                         environment)

        self._reserve_price: float = reserve_price

    def generate_reserve_price(self, load: 'Load', node: 'Node') -> float:  # this should be a float
        """
        To be called by the Nodes before an auction
        """
        return self._reserve_price  # Yes it is a lot ;)
