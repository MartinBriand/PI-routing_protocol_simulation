"""
The prj_typing package
"""

from tensorflow.python.framework.ops import EagerTensor
from typing import TYPE_CHECKING, TypeVar, Dict, Callable, Tuple

if TYPE_CHECKING:
    from Mechanics.Actors.nodes.node import Node
    from Mechanics.Actors.carriers.carrier import Carrier
    from Mechanics.Tools.load import Load


# for Carrier
CarrierBid = Dict['Node', float]
CarrierState = TypeVar(EagerTensor)

# for Node
NodeWeights = Dict['Node', Dict['Node', float]]

# for Shipper
Law = Callable[..., int]

# for Environment
Distance = Dict['Node', Dict['Node', int]]
NodeStates = Dict['Node', EagerTensor]

# for Auction
AuctionWeights = Dict['Load', Dict['Node', float]]
AuctionReservePrice = Dict['Load', float]
AuctionBid = Dict['Load', Dict['Carrier', Dict['Node', float]]]

# for Load
Cost = Tuple['Node', 'Node', float, float]
