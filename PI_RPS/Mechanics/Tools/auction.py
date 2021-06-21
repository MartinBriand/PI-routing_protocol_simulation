"""
Auction file
"""

import abc

import random

from typing import TYPE_CHECKING, Optional, Tuple, List, Dict, Type
from PI_RPS.prj_typing.types import MultiLaneAuctionWeights, MultiLaneAuctionBid
from PI_RPS.prj_typing.types import SingleLaneAuctionBid
from PI_RPS.prj_typing.types import AuctionReservePrice

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Tools.load import Load
    from PI_RPS.Mechanics.Actors.Carriers.carrier import Carrier


class Auction(abc.ABC):
    """An auction only has the method run called by the node"""

    def __init__(self, source: 'Node') -> None:
        self._source: 'Node' = source
        self._loads: List['Load'] = source.waiting_loads
        self._carriers: List['Carrier'] = source.waiting_carriers

        # The following dictionaries are going to be changed in the call of run
        # The data structure is described in each of the corresponding function

        self._reserve_prices: 'AuctionReservePrice' = {}
        self._results: Dict = {'loads': {}, 'Carriers': {}}

        self._source.signal_as_current_auction(self)

    @abc.abstractmethod
    def run(self) -> None:
        """The only function to be called in the auction by another class instance (namely a Nodes here)"""
        raise NotImplementedError

    def _terminate_auction(self) -> None:
        """Make an auction independent of the state of the parent node"""
        del self._loads
        del self._carriers

    def _get_reserve_price(self, load: 'Load') -> None:
        """
        Build the dictionary of reserve prices. The key is the auctioned load, the value is the reserve price
        """
        self._reserve_prices[load] = load.shipper.generate_reserve_price(load, self._source)

    def _notify_load(self, load: 'Load') -> None:
        """Notify the loads after making the attributions"""
        d = self._results['loads'][load]
        if d['is_attributed']:
            load.get_attribution(**d['kwargs'])
        else:
            load.discard()  # if later load expects something else, we can use **d['kwargs'] as an argument

    def _notify_winning_carrier(self, winning_carrier: 'Carrier') -> None:
        """Notify the Carriers after making the attribution"""
        d = self._results['Carriers'][winning_carrier]
        winning_carrier.get_attribution(**d['kwargs'])

    def _write_loosing_carriers(self) -> None:
        """Notify the remaining Carriers in the auction list that they """
        for carrier in self._carriers:
            self._results['Carriers'][carrier] = {'is_attributed': False, 'kwargs': {}}

    def _ask_payment(self, load: 'Load') -> None:
        """Ask the shipper to pay the carrier and the node"""
        d = self._results['loads'][load]['kwargs']
        load.shipper.proceed_to_payment(node=d['previous_node'],
                                        node_value=d['previous_node_cost'],
                                        carrier=d['carrier'],
                                        carrier_value=d['carrier_cost'])

    @property
    def results(self):
        return self._results


class MultiLanesAuction(Auction):
    """
    Auction for multi lane bidding carriers
    """

    def __init__(self,
                 source: 'Node') -> None:
        super().__init__(source)
        # The following dictionaries are going to be changed in the call of run
        # The data structure is described in each of the corresponding function
        self._weights: 'MultiLaneAuctionWeights' = {}
        self._bids: 'MultiLaneAuctionBid' = {}
        self._all_bids: Dict['Carrier', Dict['Node', float]] = {}

    def run(self) -> None:
        random.shuffle(self._carriers)  # No waiting list since they can decide to leave when they want
        # Don't randomize load waiting list so that we have a queue
        if len(self._loads) > 0 and len(self._carriers) > 0:
            self._get_all_bids()
        while len(self._loads) > 0 and len(self._carriers) > 0:
            load = self._loads[0]
            nb_carriers_involved = max(1, len(self._carriers) - len(self._loads) + 1)  # to keep some Carriers for later
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

    def _calculate_auction_weights(self, load: 'Load') -> None:
        """
        Build the dictionary of weights. The first key is the auctioned load, the second is the intermediary Nodes
        The value is the weight
        """
        self._weights[load] = self._source.weights[load.arrival]

    def _terminate_auction(self) -> None:
        super()._terminate_auction()
        del self._bids
        # but we keep self._all_bids

    def _get_all_bids(self):
        """Get all the bids from which _get_bids will create the correct bids"""
        for carrier in self._carriers:
            self._all_bids[carrier] = carrier.bid()

    def _get_bids(self, load: 'Load', nb_carriers_involved: int) -> None:
        """
        Build the dictionary of the bid dictionary. The first key is the load, key2 the carrier, and key3 the next
        node. The value is the bid
        """
        self._bids[load] = {}
        for carrier in self._carriers[:nb_carriers_involved]:  # could be parallelized
            self._bids[load][carrier] = self._all_bids[carrier]

    def _make_attributions_and_payments(self, load: 'Load',
                                        nb_carriers_involved: int) -> Tuple[bool, Optional['Carrier']]:
        """
        This is the auction process. It builds the result dictionary
        the first key is either 'loads' or 'Carriers'
        In the 'loads' dictionary, we have, for each load key:
            * an 'is_attributed' key with a boolean value associated
            * a 'kwargs' dictionary with the exact format of the kw of the get_attribution function of the load package
                (or an empty dictionary to call the discard function if need be)
        In the 'Carriers' dictionary, we have, for each Carriers key:
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
        winning_bid = final_bids[0]  # No out of bound error because at least one carrier
        winning_carrier, winning_next_node = winning_bid[0]
        winning_value = winning_bid[1]
        if winning_value <= this_auction_reserve_price:
            if nb_carriers_involved == 1 or this_auction_reserve_price < final_bids[1][1]:
                carrier_cost = this_auction_reserve_price - this_auction_weights[winning_next_node]
                reserve_price_involved = True
            else:
                carrier_cost = final_bids[1][1] - this_auction_weights[winning_next_node]
                reserve_price_involved = False

            self._results['loads'][load] = \
                {'is_attributed': True,
                 'kwargs': {'carrier': winning_carrier,
                            'previous_node': self._source,
                            'next_node': winning_next_node,
                            'carrier_cost': carrier_cost,
                            'previous_node_cost': self._source.auction_cost(),
                            'reserve_price_involved': reserve_price_involved},
                 'previous_node': self._source,
                 'winning_next_node': winning_next_node,
                 'winning_transformed_bid': winning_value,
                 'weight': this_auction_weights[winning_next_node],
                 'reserve_price': this_auction_reserve_price
                 }
            self._results['Carriers'][winning_carrier] = \
                {'is_attributed': True,
                 'kwargs': {'load': load,
                            'next_node': winning_next_node,
                            'reserve_price_involved': reserve_price_involved}
                 }
            return True, winning_carrier
        else:
            self._results['loads'][load] = {'is_attributed': False,
                                            'kwargs': {},
                                            'previous_node': self._source,
                                            'winning_next_node': winning_next_node,
                                            'winning_transformed_bid': winning_value,
                                            'weight': this_auction_weights[winning_next_node],
                                            'reserve_price': this_auction_reserve_price
                                            }
            return False, None


class SingleLaneAuction(Auction):
    """Auction for single lane bidding carriers"""
    def __init__(self, source):
        super().__init__(source)
        # The following dictionaries are going to be changed in the call of run
        # The data structure is described in each of the corresponding function
        self._bids: 'SingleLaneAuctionBid' = {}
        self._all_bids: Dict['Carrier', Dict['Node', float]] = {}

    def run(self) -> None:
        random.shuffle(self._carriers)  # No waiting list since they can decide to leave when they want
        # Don't randomize load waiting list so that we have a queue
        if len(self._loads) > 0 and len(self._carriers) > 0:
            self._get_all_bids()
        while len(self._loads) > 0 and len(self._carriers) > 0:
            load = self._loads[0]
            nb_carriers_involved = max(1, len(self._carriers) - len(
                self._loads) + 1)  # to keep some Carriers for later
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
        super()._terminate_auction()
        del self._bids
        # but we keep self._all_bids

    def _get_all_bids(self):
        """Get all the bids from which _get_bids will create the correct bids"""
        for carrier in self._carriers:
            self._all_bids[carrier] = carrier.bid()

    def _get_bids(self, load: 'Load', nb_carriers_involved: int) -> None:
        """
        Build the dictionary of the bid dictionary. The first key is the load, key2 the carrier, and key3 the next
        node. The value is the bid
        """
        self._bids[load] = {}
        for carrier in self._carriers[:nb_carriers_involved]:  # could be parallelized
            self._bids[load][carrier] = self._all_bids[carrier][load.arrival]

    def _make_attributions_and_payments(self, load: 'Load',
                                        nb_carriers_involved: int) -> Tuple[bool, Optional['Carrier']]:

        this_auction_reserve_price = self._reserve_prices[load]
        this_auction_bids = self._bids[load]
        final_bids = sorted(this_auction_bids.items(), key=lambda item: item[1])

        winning_bid = final_bids[0]  # No out of bound error because at least one carrier
        winning_carrier = winning_bid[0]
        winning_value = winning_bid[1]

        if winning_value <= this_auction_reserve_price:
            if nb_carriers_involved == 1 or this_auction_reserve_price < final_bids[1][1]:
                carrier_cost = this_auction_reserve_price
                reserve_price_involved = True
            else:
                carrier_cost = final_bids[1][1]
                reserve_price_involved = False

            self._results['loads'][load] = \
                {'is_attributed': True,
                 'kwargs': {'carrier': winning_carrier,
                            'previous_node': self._source,
                            'next_node': load.arrival,
                            'carrier_cost': carrier_cost,
                            'previous_node_cost': self._source.auction_cost(),
                            'reserve_price_involved': reserve_price_involved},
                 'previous_node': self._source,
                 'winning_next_node': load.arrival,
                 'winning_transformed_bid': winning_value,
                 'weight': 0.,
                 'reserve_price': this_auction_reserve_price
                 }
            self._results['Carriers'][winning_carrier] = \
                {'is_attributed': True,
                 'kwargs': {'load': load,
                            'next_node': load.arrival,
                            'reserve_price_involved': reserve_price_involved
                            }
                 }
            return True, winning_carrier
        else:
            self._results['loads'][load] = {'is_attributed': False,
                                            'kwargs': {},
                                            'previous_node': self._source,
                                            'winning_next_node': load.arrival,
                                            'winning_transformed_bid': winning_value,
                                            'weight': 0.,
                                            'reserve_price': this_auction_reserve_price
                                            }
            return False, None


available_auction_types: Dict[str, Type[Auction]] = {'MultiLanes': MultiLanesAuction, 'SingleLane': SingleLaneAuction}
