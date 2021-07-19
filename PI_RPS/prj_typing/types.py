"""
The prj_typing package
"""

from typing import Dict, Callable, Tuple, Union, List


# for Carrier
CarrierSingleBid = float
CarrierMultiBid = Dict['Node', float]
CarrierBid = Union[CarrierMultiBid, CarrierSingleBid]

# for Node
NodeWeights = Dict['Node', Dict['Node', float]]

# for Shipper
Law = Callable[..., None]  # is supposed to generate loads

# for Environment
Distance = Dict['Node', Dict['Node', int]]

# for carriers
CostsTable = Dict['Node', float]
ListOfCostsTable = Dict['Node', List[float]]

# for Auction
AuctionReservePrice = Dict['Load', float]

MultiLaneAuctionWeights = Dict['Load', Dict['Node', float]]
MultiLaneAuctionBid = Dict['Load', Dict['Carrier', Dict['Node', float]]]

SingleLaneAuctionBid = Dict['Load', Dict['Carrier', float]]

# for Load
Movement = Tuple['Node', 'Node', 'Carrier', float, float, bool]

# class
AllLearningCarrier = Union['LearningCarrier', 'EpisodeLearningCarrier']
