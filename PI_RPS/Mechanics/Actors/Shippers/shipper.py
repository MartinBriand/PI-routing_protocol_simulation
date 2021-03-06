"""
Shipper file
"""

import abc
from PI_RPS.Mechanics.Tools.load import Load

from typing import TYPE_CHECKING, List, Dict, Any
from PI_RPS.prj_typing.types import Law

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Actors.Carriers.carrier import Carrier
    from PI_RPS.Mechanics.Environment.environment import Environment


class Shipper(abc.ABC):
    """
    A Shipper is able to generate goods according to a law, generate reserve prices each time one of its good is
    auctioned at a Node, and has to pay the Nodes and the Carriers
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
        To be called by an auction before running
        """
        raise NotImplementedError

    def proceed_to_payment(self,
                           node: 'Node',
                           node_value: float,
                           carrier: 'Carrier',
                           carrier_value: float) -> None:
        """
        To be called by the auction after running
        """
        node.receive_payment(node_value)
        carrier.receive_payment(carrier_value)
        total_value = carrier_value + node_value
        self._expenses.append(total_value)
        self._total_expenses += total_value

    def add_load(self, load: Load) -> None:
        """Called by the load at creation to signal to the shipper"""
        self._loads.append(load)

    def clear_loads(self) -> None:
        """Called by the environment for memory cleaning"""
        self._loads.clear()

    def clear_expenses(self) -> None:
        """Called by the environment for memory cleaning"""
        self._expenses.clear()
        self._total_expenses = 0

    def add_law(self, law: 'NodeLaw'):
        """Called during initialization to add law to law list"""
        assert law.owner == self, "Please add only laws whose owner is self"
        self._laws.append(law)


class NodeLaw:
    """
    A law is an association of a Nodes and a statistical law.
    The only method is a call function to generate a number of load
    """

    def __init__(self,
                 owner: Shipper,
                 law: Law,
                 params: Dict[str, Any]) -> None:

        self._owner: Shipper = owner
        self._law: Law = law
        self._params = params

    def call(self) -> None:
        """Calling the law to generate loads"""
        self._law(**self._params)

    @property
    def owner(self) -> Shipper:
        return self._owner
