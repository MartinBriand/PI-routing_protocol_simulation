"""
The most basic shipper you could think of
"""

from PI_RPS.Mechanics.Actors.Shippers.shipper import Shipper
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Tools.load import Load
    from PI_RPS.Mechanics.Actors.Shippers.shipper import NodeLaw
    from PI_RPS.Mechanics.Environment.environment import Environment


class DummyShipper(Shipper):

    def __init__(self,
                 name: str,
                 laws: List['NodeLaw'],  # forward reference
                 expenses: List[float],
                 loads: List['Load'],
                 environment: 'Environment',
                 reserve_price_per_distance: float,
                 default_reserve_price: float) -> None:

        super().__init__(name,
                         laws,  # forward reference
                         expenses,
                         loads,
                         environment)

        self._reserve_price_per_distance: float = reserve_price_per_distance
        self._default_reserve_price: float = default_reserve_price

    def generate_reserve_price(self, load: 'Load', node: 'Node') -> float:  # this should be a float
        """
        To be called by the Nodes before an auction
        """
        if self._environment.default_reserve_price:
            return self._default_reserve_price
        else:
            return self._reserve_price_per_distance * self._environment.get_distance(node, load.arrival)
