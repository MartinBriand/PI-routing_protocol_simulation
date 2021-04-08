"""
Shipper file
"""
from Game.load import Load


class Shipper:
    """
    A Shipper is able to generate goods according to a law, generate reserve prices for each time one of its good is
    auctioned at a node, and has to pay the nodes and the carriers
    """

    def __init__(self, laws, expenses=None, loads=None, environment=None):

        if loads is None:
            loads = []
        if expenses is None:
            expenses = []

        self.environment = environment
        self.laws = laws
        self.expenses = expenses
        self.total_expenses = sum(self.expenses)
        self.loads = loads

    def generate_loads(self):
        """
        To be called by the environment at each new round to generate new loads
        """
        new_loads = []

        for law in self.laws:
            departure_node = law.departure_node  # tolereance for not writing a getter method
            arrival_node = law.arrival_node  # idem
            n = law.call()
            for k in range(n):
                new_loads.append(Load(departure_node, arrival_node, self, self.environment))

        return new_loads

    def generate_reserve_price(self, load, node):
        """
        To be called by the node before an auction
        """
        raise NotImplementedError

    def proceed_to_payment_node(self, node, node_value, carrier, carrier_value):
        """
        To be called by the node after an auction
        """
        node.receive_payment(self, node_value)  # TODO: implement this function
        carrier.receive_payment(self, node, carrier_value)
        total_value = carrier_value + node_value
        self.expenses.append(total_value)
        self.total_expenses += total_value


class NodeLaw:
    """
    A law is an association of a node and a statistical law.
    The only method is a call function to generate a number of load to be created by the shipper at a specific node
    """

    def __init__(self, departure_node, arrival_node, law, params):
        """
        The node is just the node reference
        The law should be a numpy.random.Generator.law
        The params should be the parameters to be called by the law
        """
        self.departure_node = departure_node
        self.arrival_node = arrival_node
        self._law = lambda: law(*params)

    def call(self):
        return self._law()
