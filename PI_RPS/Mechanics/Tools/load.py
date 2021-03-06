"""
Load file
"""

from typing import TYPE_CHECKING, Optional, List
from PI_RPS.prj_typing.types import Movement

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Actors.Shippers.shipper import Shipper
    from PI_RPS.Mechanics.Actors.Carriers.carrier import Carrier
    from PI_RPS.Mechanics.Environment.environment import Environment


class Load:
    """
    A load tracks what is happening to it. It has a Shippers, a current Carriers, and a route with its associated costs
    """

    def __init__(self,
                 departure: 'Node',
                 arrival: 'Node',
                 shipper: 'Shipper',
                 environment: 'Environment') -> None:

        self._departure: 'Node' = departure
        self._arrival: 'Node' = arrival
        self._shipper: 'Shipper' = shipper
        self._environment: 'Environment' = environment

        self._current_carrier: Optional['Carrier'] = None
        self._next_node: 'Node' = self._departure  # note that if you are not in transit you are at a Nodes,
        # and your next_node is also your current node then

        self._in_transit: bool = False
        self._is_arrived: bool = False
        self._is_discarded: bool = False
        self._has_new_infos: bool = False

        self._movements: List[Movement] = []
        # movement is tuple (previous_node, next_node, carrier, carrier_cost, previous_node_cost)
        self._previous_infos: List['Info'] = [Info(self._departure, self._departure, 0)]
        # to calculate the new info, use the
        # old info and add the cost of the current step

        # And now add it to the departure Nodes and the environment
        self._shipper.add_load(self)
        self._departure.add_load_to_waiting_list(self)
        self._environment.add_load(self)

    def get_attribution(self,
                        carrier: 'Carrier',
                        previous_node: 'Node',
                        next_node: 'Node',
                        carrier_cost: float,
                        previous_node_cost: float,
                        reserve_price_involved: bool) -> None:
        """
        To be called by the Node each time a load which was waiting at a Node get attributed for a next hop
        """
        self._in_transit = True
        self._current_carrier = carrier
        self._next_node.remove_load_from_waiting_list(self)
        self._next_node = next_node
        self._movements.append((previous_node,
                                next_node,
                                carrier,
                                carrier_cost,
                                previous_node_cost,
                                reserve_price_involved))

        self._new_node_infos(next_node, carrier_cost + previous_node_cost)  # we call the new infos function at each
        # attribution

    def _new_node_infos(self,
                        next_node: 'Node',
                        cost: float) -> None:
        """
        Generates new info after the attribution and tells the environment it has new info
        """
        infos = []
        if len(self._previous_infos) >= self._environment.max_nb_infos_per_load:
            # Avoid recording ping-pong
            self._previous_infos = self._previous_infos[1-self._environment.max_nb_infos_per_load:]
        for info in self._previous_infos:
            infos.append(Info(info.start, next_node, info.cost + cost))
            # Even if you go back to yourself, it is important to have the info

        infos.append(Info(next_node, next_node, 0))

        self._previous_infos = infos
        self._has_new_infos = True
        self._environment.add_load_to_new_infos_list(self)

    def communicate_infos(self) -> List['Info']:  # forward reference (hence the '')
        """Communicate the new info to the environment when asked to"""
        self._has_new_infos = False
        return self._previous_infos

    def arrive_at_next_node(self) -> None:
        """
        to be called by the Carriers each time it arrives at a next Nodes
        """
        self._current_carrier = None
        self._in_transit = False
        if self._next_node == self._arrival:
            self._is_arrived = True
        else:
            self._next_node.add_load_to_waiting_list(self)

    def discard(self) -> None:
        """
        Set the load as discarded. Called by the Nodes when auction run but no result
        (A Shippers could eventually also call it but it is not implemented yet (and will probably not be))
        """
        self._is_discarded = True
        self._next_node.remove_load_from_waiting_list(self)

    def has_new_infos(self) -> bool:
        """access the has new infos variable. Called by the environment"""
        return self._has_new_infos

    def total_delivery_cost(self) -> float:
        """Return the total delivery cost of the load (so far)"""
        return sum([element[3] for element in self._movements]) + sum([element[4] for element in self._movements])

    def nb_hops(self) -> int:
        """Returns the number of hops of the load (so far)"""
        return len(self._movements)

    def delivery_time(self) -> int:
        """Return the delivery time of the load (so far)"""
        return sum([self._environment.get_distance(departure, arrival)
                    for departure, arrival, _, _, _, _ in self._movements])

    @property
    def departure(self) -> 'Node':
        return self._departure

    @property
    def arrival(self) -> 'Node':
        return self._arrival

    @property
    def shipper(self) -> 'Shipper':
        return self._shipper

    @property
    def is_arrived(self) -> bool:
        return self._is_arrived

    @property
    def is_discarded(self) -> bool:
        return self._is_discarded

    @property
    def in_transit(self) -> bool:
        return self._in_transit

    @property
    def movements(self) -> List[Movement]:
        return self._movements


class Info:
    """
    An info is made of a start node, an arrival node, and a cost between the two
    """

    def __init__(self,
                 start: 'Node',
                 arrival: 'Node',
                 cost: float) -> None:

        self._start: 'Node' = start
        self._arrival: 'Node' = arrival
        self._cost: float = cost

    @property
    def start(self) -> 'Node':
        return self._start

    @property
    def arrival(self) -> 'Node':
        return self._arrival

    @property
    def cost(self) -> float:
        return self._cost
