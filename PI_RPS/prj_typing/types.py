"""
The prj_typing package
"""

from tensorflow.python.framework.ops import EagerTensor
from typing import Dict, Callable, Tuple, Union, List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Carriers.episode_learning_carrier import EpisodeLearningCarrier
    from PI_RPS.Mechanics.Actors.Carriers.learning_carrier import LearningCarrier



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
