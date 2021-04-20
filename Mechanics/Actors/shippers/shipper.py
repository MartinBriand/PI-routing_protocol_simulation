"""
Shipper file
"""
from Mechanics.Tools.load import Load
import abc
from typing import TYPE_CHECKING, Callable, List, Dict
if TYPE_CHECKING:
    from Mechanics.Actors.nodes.node import Node
    from Mechanics.Actors.carriers.carrier import Carrier
    from Mechanics.environment import Environment

Law = Callable[..., int]


class Shipper(abc.ABC):
    """
    A Shipper is able to generate goods according to a law, generate reserve prices for each time one of its good is
    auctioned at a nodes, and has to pay the nodes and the carriers
    """

    def __init__(self,
                 name: str,
                 laws: List['NodeLaw'],  # forward reference
                 expenses: List[float],
                 loads: List[Load],
                 environment: 'Environment') -> None:

        self._name: str = name
        self._environment: 'Environment' = environment
        self._laws: List['NodeLaw'] = laws
        self._expenses: List[float] = expenses
        self._total_expenses: float = sum(self._expenses)
        self._loads: List['Load'] = loads

        self._environment.add_shipper(self)

    def generate_loads(self) -> None:
        """
        To be called by the environment at each new round to generate new loads
        """

        for law in self._laws:
            n = law.call()
            for k in range(n):
                Load(law.departure_node, law.arrival_node, self, self._environment)

    @abc.abstractmethod
    def generate_reserve_price(self, load: Load, node: 'Node') -> float:
        """
        To be called by the nodes before an auction
        """

    def proceed_to_payment(self,
                           node: 'Node',
                           node_value: float,
                           carrier: 'Carrier',
                           carrier_value: float) -> None:
        """
        To be called by the auction after an auction
        """
        node.receive_payment(node_value)
        carrier.receive_payment(carrier_value)
        total_value = carrier_value + node_value
        self._expenses.append(total_value)
        self._total_expenses += total_value

    def add_load(self, load: Load) -> None:
        """called by the load to signal to the shipper"""
        self._loads.append(load)


class NodeLaw:
    """
    A law is an association of a nodes and a statistical law.
    The only method is a call function to generate a number of load to be created by the shippers at a specific nodes
    """

    def __init__(self,
                 departure_node: 'Node',
                 arrival_node: 'Node',
                 law: Law,
                 params: Dict):
        """
        The nodes is just the nodes reference
        The law should be a numpy.random.Generator.law (or anything else)
        The params should be the parameters to be called by the law
        """
        self._departure_node: 'Node' = departure_node
        self._arrival_node: 'Node' = arrival_node
        self._law: Callable[[None], int] = lambda: law(**params)

    def call(self) -> int:
        """Calling the law to generate a number"""
        return self._law()

    @property
    def departure_node(self) -> 'Node':
        return self._departure_node

    @property
    def arrival_node(self) -> 'Node':
        return self._arrival_node
