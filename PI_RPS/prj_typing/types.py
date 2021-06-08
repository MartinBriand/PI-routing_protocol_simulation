"""
The prj_typing package
"""

from tensorflow.python.framework.ops import EagerTensor
from typing import TYPE_CHECKING, Dict, Callable, Tuple, Union

if TYPE_CHECKING:
    pass


# for Carrier
CarrierSingleBid = float
CarrierMultiBid = Dict['Node', float]
CarrierBid = Union[CarrierMultiBid, CarrierSingleBid]
CarrierState = EagerTensor

# for Node
NodeWeights = Dict['Node', Dict['Node', float]]

# for Shipper
Law = Callable[..., None]  # is supposed to generate loads

# for Environment
Distance = Dict['Node', Dict['Node', int]]
NodeStates = Dict['Node', EagerTensor]

# for Auction
AuctionWeights = Dict['Load', Dict['Node', float]]
AuctionReservePrice = Dict['Load', float]
AuctionBid = Dict['Load', Dict['Carrier', Dict['Node', float]]]

# for Load
Movement = Tuple['Node', 'Node', 'Carrier', float, float, bool]
