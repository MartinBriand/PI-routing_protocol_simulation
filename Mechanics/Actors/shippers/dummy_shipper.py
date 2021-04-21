"""
The most basic shipper you could think of
"""

from Mechanics.Actors.shippers.shipper import Shipper
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Mechanics.Actors.nodes.node import Node
    from Mechanics.Tools.load import Load


class DummyShipper(Shipper):

    def generate_reserve_price(self, load: 'Load', node: 'Node') -> float:  # this should be a float
        """
        To be called by the nodes before an auction
        """
        return 300  # Yes it is a lot ;)
