"""
Shipper file
"""
from Game.load import Load


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

    def __init__(self,
                 laws: [Law],
                 expenses: [float] = None,
                 loads: [Load] = None):

        if loads is None:
            loads = []
        if expenses is None:
            expenses = []

        self.law: [Law] = laws
        self.expenses: [float] = expenses
        self.loads: [Load] = loads

    def generate_loads(self):
        pass
        # TODO: after coding the Law and the load objects
