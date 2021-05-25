"""
Node file
"""

import abc
from PI_RPS.Mechanics.Tools.auction import Auction

from typing import TYPE_CHECKING, Optional, List
from PI_RPS.prj_typing.types import NodeWeights

if TYPE_CHECKING:
    from PI_RPS.Mechanics import Carrier
    from PI_RPS.Mechanics.Tools import Load, Info
    from PI_RPS.Mechanics.Environment import Environment


class Node(abc.ABC):
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

    def __init__(self,
                 name: str,
                 weights: NodeWeights,
                 revenues: List[float],
                 environment: 'Environment') -> None:

        self._name: str = name
        self._environment: 'Environment' = environment
        self._waiting_loads: List['Load'] = []  # always initialize as an empty list (the loads signal themselves)
        self._waiting_carriers: List['Carrier'] = []  # same as waiting_loads

        self._current_auction: Optional[Auction] = None
        self._past_auctions: List[Auction] = []  # They will signal at creation

        self._revenues: List[float] = revenues
        self._total_revenues: float = sum(self._revenues)

        self._weights: NodeWeights = weights  # this is a dictionary of dictionaries. First key is FINAL nodes,
        # second key is NEXT nodes to avoid cyclic weights, we avoid having NEXT_NODE == THIS_NODE or
        # FINAL_NODE == THIS_NODE however, it is clear that we can have NEXT_NODE == FINAL_NODE
        # MUST be initialized with all the structure, because not going to be created

        self._environment.add_node(self)

    def run_auction(self) -> None:
        """Create an Auction instance and run it, called by the environment"""
        if len(self._waiting_loads) > 0 and len(self._waiting_carriers) > 0:
            Auction(self)  # the auction itself will signal to the node
            self._current_auction.run()  # auto signal a current and past auction
        for carrier in self._waiting_carriers:  # If lose the auction of if no auction, they are still in this list
            carrier.dont_get_attribution()

    @abc.abstractmethod
    def update_weights_with_new_infos(self, new_infos: List['Info']) -> None:
        """
        This is the method where the nodes has some intelligence.
        """

    def remove_carrier_from_waiting_list(self, carrier: 'Carrier') -> None:
        """To be called by carriers to be removed from auction waiting list"""
        self._waiting_carriers.remove(carrier)

    def add_carrier_to_waiting_list(self, carrier: 'Carrier') -> None:
        """To be called by carriers to be added to the auction waiting list"""
        self._waiting_carriers.append(carrier)

    def remove_load_from_waiting_list(self, load: 'Load') -> None:
        """To be called by loads to be removed from load waiting list"""
        self._waiting_loads.remove(load)

    def add_load_to_waiting_list(self, load: 'Load') -> None:
        """To be called by loads to be added to load waiting list"""
        self._waiting_loads.append(load)

    def receive_payment(self, value: float) -> None:
        """To be called by shipper (on an order from the auction) when should receive payment"""
        self._revenues.append(value)
        self._total_revenues += value

    @abc.abstractmethod
    def auction_cost(self) -> float:
        """To calculate the auction cost on a demand of the auction, before asking the shipper to pay"""

    @abc.abstractmethod
    def initialize_weights(self) -> None:
        """May not be called by always, but there in case it is needed for new games"""

    def signal_as_current_auction(self, auction: Auction) -> None:
        self._current_auction = auction

    def signal_as_past_auction(self, auction: Auction) -> None:
        self._past_auctions.append(auction)
        self._current_auction = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def waiting_loads(self) -> List['Load']:
        return self._waiting_loads

    @property
    def waiting_carriers(self) -> List['Carrier']:
        return self._waiting_carriers

    @property
    def weights(self) -> NodeWeights:
        return self._weights
