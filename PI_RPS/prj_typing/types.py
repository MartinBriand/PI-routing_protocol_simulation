"""
The prj_typing package
"""

from tensorflow.python.framework.ops import EagerTensor
from typing import TYPE_CHECKING, Dict, Callable, Tuple, Union, Type

if TYPE_CHECKING:
    pass


# for Carrier
CarrierSingleBid = Type[float]
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
AuctionReservePrice = Dict['Load', float]

MultiLaneAuctionWeights = Dict['Load', Dict['Node', float]]
MultiLaneAuctionBid = Dict['Load', Dict['Carrier', Dict['Node', float]]]

SingleLaneAuctionBid = Dict['Load', Dict['Carrier', float]]

# for Load
Movement = Tuple['Node', 'Node', 'Carrier', float, float, bool]
