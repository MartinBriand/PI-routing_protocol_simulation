"""
Node file
"""


class Node:
    """
    This is the node class. A node should:
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

    def __init__(self, past_auctions, weights):
        self.waiting_loads = []  # always initialize as an empty list since the loads add themselves to the list after
        self.waiting_carriers = []  # same as waiting_loads
        self.past_auctions = past_auctions

        self.weights = weights  # this is a dictionary of dictionaries. First key is FINAL node, second key is NEXT node
        # to avoid cyclic weights, we avoid having NEXT_NODE = THIS_NODE

    def run_auction(self):
        """Create an Auction instance and run it"""
        current_auction = Auction(self)
        current_auction.run()
        self.past_auctions.append(current_auction)

    def update_weights_with_new_infos(self, new_infos):
        """
        This is the method where the node has some intelligence
        """
        raise NotImplementedError

    def remove_carrier_from_waiting_list(self, carrier):
        self.waiting_carriers.remove(carrier)

    def add_carrier_to_waiting_list(self, carrier):
        self.waiting_carriers.append(carrier)

    def remove_load_from_waiting_list(self, load):
        self.waiting_loads.remove(load)

    def add_load_from_waiting_list(self, load):
        self.waiting_loads.append(load)


class Auction:
    def __init__(self, source):
        self.source = source
        self.loads = source.waiting_loads
        self.carriers = source.waiting_carriers

        # The following four dictionaries are going to be changed in the call of run
        # The data structure is described in each of the corresponding function
        self.weights = {}
        self.reserve_prices = {}
        self.bids = {}
        self.results = {}

    def run(self):
        """The only function to be called in the auction by another class instance (namely a node here)"""
        self._calculate_auction_weights()
        self._get_reserve_prices()
        self._get_bids()
        self._make_attribution_and_payments()
        self._notify_loads()
        self._notify_carriers()

    def _calculate_auction_weights(self):
        """
        Build the dictionary of weights. The first key is the auctioned load, the second is the intermediary node
        The value is the weight
        """
        for load in self.loads:
            self.weights[load] = self.source.weights[load.arrival]

    def _get_reserve_prices(self):
        """
        Build the dictionary of reserve prices. The key is the auctioned load, the value is the reserve price
        """
        for load in self.loads:
            self.reserve_prices[load] = load.shipper.generate_reserve_price(load, self.source)

    def _get_bids(self):
        """
        Build the dictionary of the bid dictionary. The first key is the carrier, the second is the next node,
        the value is the bid
        """
        for carrier in self.carriers:
            self.bids[carrier] = carrier.bid(self.source)

    def _notify_loads(self):
        """Notify the loads after making the attributions"""
        for load in self.results['loads']:
            d = self.results['loads'][load]
            if d['is_attributed']:
                load.get_attribution(**d['kwargs'])
            else:
                load.discard()

    def _notify_carriers(self):
        """Notify the carriers after making the attribution"""
        for carrier in self.results['carriers']:
            d = self.results['carriers'][carrier]
            if d['is_attributed']:
                carrier.get_attribution(**d['kwargs'])
            else:
                carrier.dont_get_attribution()

    def _make_attribution_and_payments(self):
        """
        This is the auction process. It builds the result dictionary
        the first key is either 'loads' or 'carriers'
        In the 'loads' dictionary, we have, for each load key:
            * an 'is_attributed' key with a boolean value associated
            * a 'kwargs' dictionary with the exact format of the kw of the get_attribution function of the load package
                (or an empty dictionary to call the discard function if need be)
        In the 'carriers' dictionary, we have, for each carrier key:
            * an 'is_attributed' key with a boolean value associated
            * a 'kwargs' dictionary with the exact format of the kw of the get_attribution function of the load package
                (or an empty dictionary to call the dont_get_attribution function if need be)
        """
        pass


