"""
Auction file
"""

import random
from typing import TYPE_CHECKING, Optional, Tuple, List, Dict
if TYPE_CHECKING:
    from Mechanics.Actors.nodes.node import Node
    from Mechanics.Tools.load import Load
    from Mechanics.Actors.carriers.carrier import Carrier

AuctionWeights = Dict['Load', Dict['Node', float]]
AuctionReservePrice = Dict['Load', float]
AuctionBid = Dict['Load', Dict['Carrier', Dict['Node', float]]]


class Auction:
    """
    An auction correspond to one auction being run
    """

    def __init__(self,
                 source: 'Node') -> None:

        self._source: 'Node' = source
        self._loads: List['Load'] = source.waiting_loads
        self._carriers: List['Carrier'] = source.waiting_carriers

        # The following four dictionaries are going to be changed in the call of run
        # The data structure is described in each of the corresponding function
        self._weights: 'AuctionWeights' = {}
        self._reserve_prices: 'AuctionReservePrice' = {}
        self._bids: 'AuctionBid' = {}
        self._results: Dict = {'loads': {}, 'carriers': {}}

        self._source.signal_as_current_auction(self)

    def run(self) -> None:
        """The only function to be called in the auction by another class instance (namely a nodes here)"""
        random.shuffle(self._carriers)  # No waiting list since they can decide to leave when they want
        # Don't randomize load waiting list so that we have a queue
        while len(self._loads) > 0 and len(self._carriers) > 0:
            load = self._loads[0]
            nb_carriers_involved = max(1, len(self._carriers) - len(self._loads) + 1)  # to keep some carriers for later
            self._calculate_auction_weights(load)
            self._get_reserve_price(load)
            self._get_bids(load, nb_carriers_involved)
            load_attributed, winning_carrier = self._make_attributions_and_payments(load, nb_carriers_involved)
            self._notify_load(load)
            if load_attributed:  # This step is important to make sure they do not participate in the next auction
                self._notify_winning_carrier(winning_carrier)
                self._ask_payment(load)

        self._write_loosing_carriers()
        self._terminate_auction()
        self._source.signal_as_past_auction(self)

    def _terminate_auction(self) -> None:
        """Make an auction independent of the state of the parent node"""
        del self._loads
        del self._carriers

    def _calculate_auction_weights(self, load: 'Load') -> None:
        """
        Build the dictionary of weights. The first key is the auctioned load, the second is the intermediary nodes
        The value is the weight
        """
        self._weights[load] = self._source.weights[load.arrival]

    def _get_reserve_price(self, load: 'Load') -> None:
        """
        Build the dictionary of reserve prices. The key is the auctioned load, the value is the reserve price
        """
        self._reserve_prices[load] = load.shipper.generate_reserve_price(load, self._source)

    def _get_bids(self, load: 'Load', nb_carriers_involved: int) -> None:
        """
        Build the dictionary of the bid dictionary. The first key is the load, key2 the carrier, and key3 the next
        node. The value is the bid
        """
        self._bids[load] = {}
        for carrier in self._carriers[:nb_carriers_involved]:
            self._bids[load][carrier] = carrier.bid(self._source)

    def _notify_load(self, load: 'Load') -> None:
        """Notify the loads after making the attributions"""
        d = self._results['loads'][load]
        if d['is_attributed']:
            load.get_attribution(**d['kwargs'])
        else:
            load.discard()  # if later load expects something else, we can use **d['kwargs'] as an argument

    def _notify_winning_carrier(self, winning_carrier: 'Carrier') -> None:
        """Notify the carriers after making the attribution"""
        d = self._results['carriers'][winning_carrier]
        winning_carrier.get_attribution(**d['kwargs'])

    def _write_loosing_carriers(self) -> None:
        """Notify the remaining carriers in the auction list that they """
        for carrier in self._carriers:
            self._results['carriers'][carrier] = {'is_attributed': False, 'kwargs': {}}

    def _ask_payment(self, load: 'Load') -> None:
        """Ask the shipper to pay the carrier and the node"""
        d = self._results['loads'][load]['kwargs']
        load.shipper.proceed_to_payment(node=d['previous_node'],
                                        node_value=d['previous_node_cost'],
                                        carrier=d['carrier'],
                                        carrier_value=d['carrier_cost'])

    def _make_attributions_and_payments(self, load: 'Load',
                                        nb_carriers_involved: int) -> Tuple[bool, Optional['Carrier']]:
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
        this_auction_weights = self._weights[load]
        this_auction_reserve_price = self._reserve_prices[load]
        this_auction_bids = self._bids[load]

        new_bids = {}
        for carrier in this_auction_bids:
            new_bids[carrier] = {}
            for node in this_auction_bids[carrier]:
                new_bids[carrier][node] = this_auction_bids[carrier][node] + this_auction_weights[node]

        final_bids = {}
        for carrier in new_bids:
            node_arg_min = min(new_bids[carrier], key=new_bids[carrier].get)
            final_bids[(carrier, node_arg_min)] = new_bids[carrier][node_arg_min]

        final_bids = sorted(final_bids.items(), key=lambda item: item[1])
        winning_bid = final_bids[0]  # No out of bound error because
        winning_carrier, winning_next_node = winning_bid[0]
        winning_value = winning_bid[1]
        if winning_value <= this_auction_reserve_price:
            carrier_cost = min(this_auction_reserve_price, final_bids[1][1]) if nb_carriers_involved > 1 else \
                this_auction_reserve_price
            carrier_cost -= this_auction_weights[winning_next_node]
            self._results['loads'][load] = \
                {'is_attributed': True,
                 'kwargs': {'carrier': winning_carrier,
                            'previous_node': self._source,
                            'next_node': winning_next_node,
                            'carrier_cost': carrier_cost,
                            'previous_node_cost': self._source.auction_cost()}}
            self._results['carriers'][winning_carrier] = \
                {'is_attributed': True,
                 'kwargs': {'load': load, 'next_node': winning_next_node}}
            return True, winning_carrier
        else:
            self._results['loads'][load] = {'is_attributed': False, 'kwargs': {}}
            return False, None
