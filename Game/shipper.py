"""
Shipper file
"""


class Law:
    """
    A law is an association of a node and a statistical law.
    """

    pass
    # TODO: implement the Law class


class Shipper:
    """
    A Shipper is able to generate goods according to a law, generate reserve prices for each time one of its good is
    auctioned at a node, and has to pay the nodes and the carriers
    """

    def __init__(self, laws, expenses=None, loads=None):

        if loads is None:
            loads = []
        if expenses is None:
            expenses = []

        self.law = laws
        self.expenses = expenses
        self.loads = loads

    def generate_loads(self):
        pass
        # TODO: after coding the Law and the load objects
