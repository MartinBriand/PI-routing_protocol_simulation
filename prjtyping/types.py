"""
The prjtyping package
"""

from tensorflow import Tensor
from typing import Dict, Callable, Tuple


# for Carrier
CarrierBid = Dict['Node', float]

# for Node
NodeWeights = Dict['Node', Dict['Node', float]]
NodeState = Dict['Node', Tensor]

# for Shipper
Law = Callable[..., int]

# for Environment
Distance = Dict['Node', Dict['Node', int]]

# for Auction
AuctionWeights = Dict['Load', Dict['Node', float]]
AuctionReservePrice = Dict['Load', float]
AuctionBid = Dict['Load', Dict['Carrier', Dict['Node', float]]]

# for Load
Cost = Tuple['Node', 'Node', float, float]
