"""
Environment file
"""

from typing import TYPE_CHECKING, List, Optional

from PI_RPS.prj_typing.types import Distance

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Carriers.carrier import Carrier
    from PI_RPS.Mechanics.Tools.load import Load
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Actors.Shippers.shipper import Shipper


class Environment:
    """
    This is the Environment class. It should be seen as a simple clock necessary for the functioning of the game,
    but not as a real entity of the game. If a version ever has to be implemented,
    the Environment will not exist.
    """

    def __init__(self,
                 nb_hours_per_time_unit: float,
                 max_nb_infos_per_load: int,
                 init_node_weights_distance_scaling_factor: float,
                 max_node_weights_distance_scaling_factor: float,
                 t_c_mu: float,
                 t_c_sigma: float,
                 ffh_c_mu: float,
                 ffh_c_sigma: float,
                 action_min: Optional[float] = None,
                 action_max: Optional[float] = None):

        self._nb_hours_per_time_unit: float = nb_hours_per_time_unit
        self._max_nb_infos_per_load: int = max_nb_infos_per_load
        self._init_node_weights_distance_scaling_factor: float = init_node_weights_distance_scaling_factor
        self._max_node_weights_distance_scaling_factor: float = max_node_weights_distance_scaling_factor

        # for the four lists, the creation process add them to the list
        self._nodes: List[Node] = []
        self._carriers: List[Carrier] = []
        self._shippers: List[Shipper] = []
        self._loads: List[Load] = []

        self._loads_with_new_infos: List[Load] = [load for load in self._loads if load.has_new_infos()]

        self._distances: Distance = {}  # should be a dictionary of dictionaries with Nodes as keys

        self._default_reserve_price: bool = True

        # for the cost function (Ideally we should have a subclass in which that is defined, because this is only for
        # CarriersWithCosts but we only play with these carriers in the simulation

        self._action_min: Optional[float] = action_min
        self._action_max: Optional[float] = action_max

        self._t_c_mu: float = t_c_mu
        self._t_c_sigma: float = t_c_sigma
        self._ffh_c_mu: float = ffh_c_mu
        self._ffh_c_sigma: float = ffh_c_sigma

    def iteration(self) -> None:
        """This is the main function of the Environment class. It represents the operation of the game for one unit of
        time. It should be called in a loop after initializing the game."""

        self._get_new_loads()
        self._run_auctions()
        self._carriers_next_states()
        self._get_and_broadcast_new_infos()

    def _get_new_loads(self) -> None:
        """Gathering new loads generated by Shippers"""
        for shipper in self._shippers:  # could be parallelized
            shipper.generate_loads()

    def _run_auctions(self) -> None:
        """Ask Nodes to run the auctions (collect bids and reserve prices, make attribution and ask for payments)"""
        for node in self._nodes:  # could be parallelized
            node.run_auction()

    def _carriers_next_states(self) -> None:
        """Asking Carriers to move"""
        for carrier in self._carriers:
            carrier.next_step()  # could be parallelized

    def _get_and_broadcast_new_infos(self) -> None:
        """Asking loads with new infos to communicate them and then broadcast the infos to Nodes"""
        new_infos = []
        for load in self._loads_with_new_infos:
            new_infos += load.communicate_infos()  # could be parallelized
        for node in self._nodes:  # optimize in the node structure with master implementation (see DummyNode for ex)
            node.update_weights_with_new_infos(new_infos)
        self._loads_with_new_infos = []

    def get_distance(self, departure: 'Node', arrival: 'Node') -> int:
        """To know the distance (called by carriers when moving or by shippers when generating reserve price
        or by nodes at initialization"""
        return self._distances[departure][arrival]

    def set_distances(self, distances: 'Distance') -> None:
        """Set distance function"""
        self._distances = distances

    def add_node(self, node: 'Node') -> None:
        """add_node function"""
        self._nodes.append(node)

    def add_carrier(self, carrier: 'Carrier') -> None:
        """add_carrier function"""
        self._carriers.append(carrier)

    def remove_carrier(self, carrier: 'Carrier') -> None:
        """Remove a carrier from the list (called by cost_learning carriers after bankruptcy)"""
        self._carriers.remove(carrier)

    def add_shipper(self, shipper: 'Shipper') -> None:
        """add_shipper function"""
        self._shippers.append(shipper)

    def add_load(self, load: 'Load') -> None:
        """add_load function"""
        self._loads.append(load)

    def add_load_to_new_infos_list(self, load: 'Load') -> None:
        """to be called by loads with new info to signal the new information"""
        self._loads_with_new_infos.append(load)

    def clear_node_auctions_and_profit(self) -> None:
        """Called by training loop before tests"""
        for node in self._nodes:
            node.clear_past_auctions()
            node.clear_profit()

    def clear_loads(self) -> None:
        """Called by training loop before tests"""
        self._loads.clear()
        self._loads_with_new_infos.clear()
        for carrier in self._carriers:
            carrier.clear_load()
        for node in self._nodes:
            node.clear_waiting_loads()
        for shipper in self._shippers:
            shipper.clear_loads()

    def clear_carrier_profits(self) -> None:
        """Called by training loop before tests"""
        for carrier in self._carriers:
            carrier.clear_profit()

    def clear_shipper_expenses(self) -> None:
        """Called by training loop before tests"""
        for shipper in self._shippers:
            shipper.clear_expenses()

    @property
    def nodes(self) -> List['Node']:
        return self._nodes

    @property
    def carriers(self) -> List['Carrier']:
        return self._carriers

    @property
    def shippers(self) -> List['Shipper']:
        return self._shippers

    @property
    def loads(self) -> List['Load']:
        return self._loads

    @property
    def nb_hours_per_time_unit(self) -> float:
        return self._nb_hours_per_time_unit

    @property
    def action_min(self) -> float:
        assert self._action_min is not None, "can't call this property if None"
        return self._action_min

    @property
    def action_max(self) -> float:
        assert self._action_max is not None, "can't call this property if None"
        return self._action_max

    @property
    def t_c_mu(self) -> float:
        return self._t_c_mu

    @property
    def t_c_sigma(self) -> float:
        return self._t_c_sigma

    @property
    def ffh_c_mu(self) -> float:
        return self._ffh_c_mu

    @property
    def ffh_c_sigma(self) -> float:
        return self._ffh_c_sigma

    @property
    def max_nb_infos_per_load(self) -> int:
        return self._max_nb_infos_per_load

    @property
    def init_node_weights_distance_scaling_factor(self) -> float:
        return self._init_node_weights_distance_scaling_factor

    @property
    def max_node_weights_distance_scaling_factor(self) -> float:
        return self._max_node_weights_distance_scaling_factor

    @property
    def default_reserve_price(self) -> bool:
        return self._default_reserve_price

    @default_reserve_price.setter
    def default_reserve_price(self, value: bool) -> None:
        assert type(value) == bool
        assert value == (not self._default_reserve_price)
        self._default_reserve_price = value
