"""
Auction file
"""

import random


class Auction:
    """
    An auction correspond to one auction being run
    """

    def __init__(self, source):
        self.source = source
        self.loads = source.waiting_loads
        self.original_loads = self.loads.copy()
        self.carriers = source.waiting_carriers
        self.original_carriers = self.carriers.copy()

        # The following four dictionaries are going to be changed in the call of run
        # The data structure is described in each of the corresponding function
        self.weights = {}
        self.reserve_prices = None
        self.bids = {}
        self.results = {'loads': {}, 'carriers': {}}

        self.source.signal_as_current_auction()

    def run(self):
        """The only function to be called in the auction by another class instance (namely a nodes here)"""
        nb_load = len(self.loads)
        random.shuffle(self.carriers)  # No waiting list since they can decide to leave when they want
        # Don't randomize load waiting list so that we have a queue
        for k in range(nb_load):
            load = self.loads[k]
            nb_carriers_involved = max(1, len(self.carriers) - nb_load + k + 1)  # self.carriers has at least 1 element
            if len(self.carriers) > 0:
                self._calculate_auction_weights(load)
                self._get_reserve_price(load)
                self._get_bids(load, nb_carriers_involved)
                load_attributed, winning_carrier = self._make_attributions_and_payments(load, nb_carriers_involved)
                self._notify_load(load)
                if load_attributed:  # This step is important to make sure they do not participate in the next auction
                    self._notify_winning_carrier(winning_carrier)
                    self._ask_payment(load)
            else:
                break  # keep the other loads in the waiting list for the next round
        self._notify_loosing_carriers()
        self._terminate_auction()
        self.source.signal_as_past_auction()

    def _terminate_auction(self):
        """Make an auction independent of the state of the parent node"""
        del self.loads
        del self.carriers

    def _calculate_auction_weights(self, load):
        """
        Build the dictionary of weights. The first key is the auctioned load, the second is the intermediary nodes
        The value is the weight
        """
        self.weights[load] = self.source.weights[load.arrival]

    def _get_reserve_price(self, load):
        """
        Build the dictionary of reserve prices. The key is the auctioned load, the value is the reserve price
        """
        self.reserve_prices[load] = load.shipper.generate_reserve_price(load, self.source)

    def _get_bids(self, load, nb_carriers_involved):
        """
        Build the dictionary of the bid dictionary. The first key is the load, key2 the carrier, and key3 the next
        node. The value is the bid
        """
        self.bids[load] = {}
        for carrier in self.carriers[:nb_carriers_involved]:
            self.bids[load][carrier] = carrier.bid(self.source)

    def _notify_load(self, load):
        """Notify the loads after making the attributions"""
        d = self.results['loads'][load]
        if d['is_attributed']:
            load.get_attribution(**d['kwargs'])
        else:
            load.discard(**d['kwargs'])

    def _notify_winning_carrier(self, winning_carrier):
        """Notify the carriers after making the attribution"""
        d = self.results['carriers'][winning_carrier]
        assert d['is_attributed'], 'winning is not winning...'
        winning_carrier.get_attribution(**d['kwargs'])

    def _notify_loosing_carriers(self):
        """Notify the remaining carriers in the auction list that they """
        for carrier in self.carriers:
            self.results['carriers'][carrier] = {'is_attributed': False, 'kwargs': {}}
        for carrier in self.carriers.copy():
            carrier.dont_get_attribution(**self.results['carriers'][carrier]['kwargs'])

    def _ask_payment(self, load):
        """Ask the shipper to pay the carrier and the node"""
        d = self.results['loads'][load]
        load.shipper.proceed_to_payment(node=d['previous_node'],
                                        node_value=d['previous_node_cost'],
                                        carrier=d['carrier'],
                                        carrier_value=d['carrier_cost'])

    def _make_attributions_and_payments(self, load, nb_carriers_involved):
        """
        This is the auction process. It builds the result dictionary
        the first key is either 'loads' or 'carriers'
        In the 'loads' dictionary, we have, for each load key:
            * an 'is_attributed' key with a boolean value associated
            * a 'kwargs' dictionary with the exact format of the kw of the get_attribution function of the load package
                (or an empty dictionary to call the discard function if need be)
        In the 'carriers' dictionary, we have, for each carriers key:
            * an 'is_attributed' key with a boolean value associated
            * a 'kwargs' dictionary with the exact format of the kw of the get_attribution function of the load package
                (or an empty dictionary to call the dont_get_attribution function if need be)
        """
        this_auction_weights = self.weights[load]
        this_auction_reserve_price = self.reserve_prices[load]
        this_auction_bids = self.bids[load]

        new_bids = {}
        for carrier in this_auction_bids:
            new_bids[carrier] = {}
            for node in this_auction_bids[carrier]:
                new_bids[(carrier, node)] = this_auction_bids[carrier][node] + this_auction_weights[node]

        new_bids = sorted(new_bids.items(), key=lambda item: item[1])
        winning_bid = new_bids[0]  # No out of bound error because
        winning_carrier, winning_next_node = winning_bid[0]
        winning_value = winning_bid[1]
        if winning_value <= this_auction_reserve_price:
            carrier_cost = min(this_auction_reserve_price, new_bids[1][1]) if nb_carriers_involved <= 1 else \
                this_auction_reserve_price
            self.results['loads'][load] = \
                {'is_attributed': True,
                 'kwargs': {'carrier': winning_carrier,
                            'previous_node': self.source,
                            'next_node': winning_next_node,
                            'carrier_cost': carrier_cost,
                            'previous_node_cost': self.source.auction_cost()}}
            self.results['carriers'][winning_carrier] = \
                {'is_attributed': True,
                 'kwargs': {'load': load, 'next_node': winning_next_node}}
            return True, winning_carrier
        else:
            self.results['loads'][load] = {'is_attributed': False, 'kwargs': {}}
            return False, None
