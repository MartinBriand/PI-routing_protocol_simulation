"""
Environment file
"""

from typing import TYPE_CHECKING, List
from prj_typing.types import Distance

if TYPE_CHECKING:
    from Mechanics.Actors.carriers.carrier import Carrier
    from Mechanics.Tools.load import Load
    from Mechanics.Actors.nodes.node import Node
    from Mechanics.Actors.shippers.shipper import Shipper


class Environment:
    """
    This is the Environment class. It should be seen as a simple clock necessary for the functioning of the game,
    but not as a real entity of the game. If a version ever have to be implemented or become more realistic,
    the Environment should be deleted.
    """

    def __init__(self):

        # for the four lists, the creation process add them to the list
        self._nodes: List[Node] = []
        self._carriers: List[Carrier] = []
        self._shippers: List[Shipper] = []
        self._loads: List[Load] = []

        self._loads_with_new_infos: List[Load] = [load for load in self._loads if load.has_new_infos()]

        self._distances: Distance = {}  # should be a dictionary of dictionaries with nodes as keys

    def iteration(self) -> None:
        """This is the main function of the Environment class. It represents the operation of the game for one unit of
        time. It should be called in a loop after initializing the game."""

        self._get_new_loads()
        self._run_auctions()
        self._carriers_next_states()
        self._get_and_broadcast_new_infos()

    def _get_new_loads(self) -> None:
        """Gathering new loads generated by shippers"""
        for shipper in self._shippers:
            shipper.generate_loads()

    def _run_auctions(self) -> None:
        """Ask nodes to run the auctions (collect bids and reserve prices, make attribution and ask for payments)"""
        for node in self._nodes:
            node.run_auction()

    def _carriers_next_states(self) -> None:
        """Asking carriers to move"""
        for carrier in self._carriers:
            carrier.next_step()

    def _get_and_broadcast_new_infos(self) -> None:
        """Asking loads with new infos to communicate this and then broadcast the information to nodes"""
        new_infos = []
        for load in self._loads_with_new_infos:
            new_infos += load.communicate_infos()
        for node in self._nodes:  # FIXME: This can be clearly optimized
            node.update_weights_with_new_infos(new_infos)
        self._loads_with_new_infos = []

    def get_distance(self, start: 'Node', end: 'Node') -> int:
        """to be called by carriers to know the remaining time"""
        return self._distances[start][end]

    def set_distance(self, distances: 'Distance') -> None:
        """the set distance function"""
        self._distances = distances

    def add_node(self, node: 'Node') -> None:
        """add_node function"""
        self._nodes.append(node)

    def add_carrier(self, carrier: 'Carrier') -> None:
        """add_carrier function"""
        self._carriers.append(carrier)

    def add_shipper(self, shipper: 'Shipper') -> None:
        """add_shipper function"""
        self._shippers.append(shipper)

    def add_load(self, load: 'Load') -> None:
        """add_load function"""
        self._loads.append(load)

    def add_load_to_new_infos_list(self, load: 'Load') -> None:
        """to be called by load with new info to signal the new information"""
        self._loads_with_new_infos.append(load)

    @property
    def nodes(self) -> List['Node']:
        return self._nodes
