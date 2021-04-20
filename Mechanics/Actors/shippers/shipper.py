"""
Shipper file
"""
from Mechanics.Tools.load import Load
import abc


class Shipper(abc.ABC):
    """
    A Shipper is able to generate goods according to a law, generate reserve prices for each time one of its good is
    auctioned at a nodes, and has to pay the nodes and the carriers
    """

    def __init__(self, name, laws, expenses, loads, environment):
        self._name = name
        self._environment = environment
        self._laws = laws
        self._expenses = expenses
        self._total_expenses = sum(self._expenses)
        self._loads = loads

        self._environment.add_shipper(self)

    def generate_loads(self):
        """
        To be called by the environment at each new round to generate new loads
        """

        for law in self._laws:
            n = law.call()
            for k in range(n):
                Load(law.departure_node, law.arrival_node, self, self._environment)

    @abc.abstractmethod
    def generate_reserve_price(self, load, node):  # this should be a float
        """
        To be called by the nodes before an auction
        """

    def proceed_to_payment(self, node, node_value, carrier, carrier_value):
        """
        To be called by the auction after an auction
        """
        node.receive_payment(node_value)
        carrier.receive_payment(carrier_value)
        total_value = carrier_value + node_value
        self._expenses.append(total_value)
        self._total_expenses += total_value

    def add_load(self, load):
        """called by the load to signal to the shipper"""
        self._loads.append(load)


class NodeLaw:
    """
    A law is an association of a nodes and a statistical law.
    The only method is a call function to generate a number of load to be created by the shippers at a specific nodes
    """

    def __init__(self, departure_node, arrival_node, law, params):
        """
        The nodes is just the nodes reference
        The law should be a numpy.random.Generator.law (or anything else)
        The params should be the parameters to be called by the law
        """
        self._departure_node = departure_node
        self._arrival_node = arrival_node
        self._law = lambda: law(**params)

    def call(self):
        """Calling the law to generate a number"""
        return self._law()

    @property
    def departure_node(self):
        return self._departure_node

    @property
    def arrival_node(self):
        return self._arrival_node
