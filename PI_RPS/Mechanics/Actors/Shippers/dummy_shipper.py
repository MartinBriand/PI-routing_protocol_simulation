"""
The most basic shipper you could think of
"""

from PI_RPS.Mechanics.Actors.Shippers.shipper import Shipper
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Tools.load import Load


class DummyShipper(Shipper):

    def generate_reserve_price(self, load: 'Load', node: 'Node') -> float:  # this should be a float
        """
        To be called by the Nodes before an auction
        """
        return 50000.  # Yes it is a lot ;)
