"""
Shipper file
"""
from Mechanics.Tools.load import Load


class Shipper:
    """
    A Shipper is able to generate goods according to a law, generate reserve prices for each time one of its good is
    auctioned at a nodes, and has to pay the nodes and the carriers
    """

    def __init__(self, name, laws, expenses, loads, environment):
        self.name = name
        self.environment = environment
        self.laws = laws
        self.expenses = expenses
        self.total_expenses = sum(self.expenses)
        self.loads = loads

        self.environment.add_shipper(self)

    def generate_loads(self):
        """
        To be called by the environment at each new round to generate new loads
        """

        for law in self.laws:
            departure_node = law.departure_node  # tolerance for not writing a getter method
            arrival_node = law.arrival_node  # idem
            n = law.call()
            for k in range(n):
                Load(departure_node, arrival_node, self, self.environment)

    def generate_reserve_price(self, load, node):  # this should be a float
        """
        To be called by the nodes before an auction
        """
        raise NotImplementedError

    def proceed_to_payment(self, node, node_value, carrier, carrier_value):
        """
        To be called by the auction after an auction
        """
        node.receive_payment(self, node_value)
        carrier.receive_payment(self, carrier_value)
        total_value = carrier_value + node_value
        self.expenses.append(total_value)
        self.total_expenses += total_value


class NodeLaw:
    """
    A law is an association of a nodes and a statistical law.
    The only method is a call function to generate a number of load to be created by the shippers at a specific nodes
    """

    def __init__(self, departure_node, arrival_node, law, params):
        """
        The nodes is just the nodes reference
        The law should be a numpy.random.Generator.law
        The params should be the parameters to be called by the law
        """
        self.departure_node = departure_node
        self.arrival_node = arrival_node
        self._law = lambda: law(*params)

    def call(self):
        """Calling the law to generate a number"""
        return self._law()
