"""
Shipper file
"""

import abc
from Mechanics.Tools.load import Load

from typing import TYPE_CHECKING, List, Dict, Any
from prj_typing.types import Law

if TYPE_CHECKING:
    from Mechanics.Actors.nodes.node import Node
    from Mechanics.Actors.carriers.carrier import Carrier
    from Mechanics.Environment.environment import Environment


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
            law.call()

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

    def add_law(self, law: 'NodeLaw'):
        """Called during initialization to add load to load list"""
        assert law.owner == self, "Please add only laws whose owner is self"
        self._laws.append(law)


class NodeLaw:
    """
    A law is an association of a nodes and a statistical law.
    The only method is a call function to generate a number of load to be created by the shippers at a specific nodes
    """

    def __init__(self,
                 owner: Shipper,
                 law: Law,
                 params: Dict[str, Any]) -> None:
        """
        Generate loads
        """
        self._owner: Shipper = owner
        self._law: Law = law
        self._params = params

    def call(self) -> None:
        """Calling the law to generate loads"""
        self._law(**self._params)

    @property
    def owner(self) -> Shipper:
        return self._owner
