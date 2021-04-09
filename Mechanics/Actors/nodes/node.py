"""
Node file
"""
from Mechanics.Tools.auction import Auction


class Node:
    """
    This is the nodes class. A nodes should:
        * Generate weights
        * Run auctions for the available loads
            * Ask shippers for reserve prices
            * Ask Carriers for bids
            * Run the auction
            * Make the attribution
            * Ask everyone to update their status
            * Ask everyone to proceed to payment
        * Make the attribution and ask for payment

    Important note: the waiting lists are only managed by the loads and the carriers themselves. A load signal when it
    wants to be auctioned, remove itself after being auctioned, and similarly for the carriers.
    """

    def __init__(self, name, past_auctions, weights, revenues, environment):
        self.name = name
        self.environment = environment
        self.waiting_loads = []  # always initialize as an empty list since the loads add themselves to the list after
        self.waiting_carriers = []  # same as waiting_loads
        self.past_auctions = past_auctions

        self.revenues = revenues
        self.total_revenues = sum(self.revenues)

        self.weights = weights  # this is a dictionary of dictionaries. First key is FINAL nodes, second key is NEXT
        # nodes to avoid cyclic weights, we avoid having NEXT_NODE == THIS_NODE or  FINAL_NODE == THIS_NODE
        # however, it is clear that we can have NEXT_NODE == FINAL_NODE
        # MUST be initialized will all the structure, because not going to be created

        self.environment.add_node(self)

    def run_auction(self):
        """Create an Auction instance and run it, called by the environment"""
        if len(self.waiting_loads) > 0 and len(self.waiting_carriers) > 0:
            current_auction = Auction(self)
            current_auction.run()
            self.past_auctions.append(current_auction)

    def update_weights_with_new_infos(self, new_infos):
        """
        This is the method where the nodes has some intelligence.
        """
        raise NotImplementedError

    def remove_carrier_from_waiting_list(self, carrier):
        """To be called by carriers to be removed from auction waiting list"""
        self.waiting_carriers.remove(carrier)

    def add_carrier_to_waiting_list(self, carrier):
        """To be called by carriers to be added to the auction waiting list"""
        self.waiting_carriers.append(carrier)

    def remove_load_from_waiting_list(self, load):
        """To be called by loads to be removed from load waiting list"""
        self.waiting_loads.remove(load)

    def add_load_from_waiting_list(self, load):
        """To be called by loads to be added to load waiting list"""
        self.waiting_loads.append(load)

    def receive_payment(self, value):
        """To be called by shipper (on an order from the auction) when should receive payment"""
        self.revenues.append(value)
        self.total_revenues += value

    def auction_cost(self):
        """To calculate the auction cost on a demand of the auction, before asking the shipper to pay"""
        raise NotImplementedError
