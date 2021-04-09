"""
The most basic shipper you could think of
"""
from Mechanics.Actors.shippers.shipper import Shipper


class DummyShipper(Shipper):

    def generate_reserve_price(self, load, node):  # this should be a float
        """
        To be called by the nodes before an auction
        """
        return 10000  # Yes it is a lot ;)
