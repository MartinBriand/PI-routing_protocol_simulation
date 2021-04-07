"""
Environment file
"""
from Game.carrier import Carrier
from Game.load import Load, Info
from Game.node import Node
from Game.shipper import Shipper


class Environment:
    """
    This is the Environment class. It should be seen as a simple clock necessary for the functioning of the game,
    but not as a real entity of the game. If a version ever have to be implemented or become more realistic,
    the Environment should be deleted.
    """

    def __init__(self,
                 carriers: [Carrier] = None,
                 shippers: [Shipper] = None,
                 nodes: [Node] = None,
                 loads: [Load] = None):

        # a few test lines to make sure we do not have mutable variables in the parameters of init
        if carriers is None:
            carriers = []
        if loads is None:
            loads = []
        if nodes is None:
            nodes = []
        if shippers is None:
            shippers = []

        # real initialization
        self.carriers: [Carrier] = carriers
        self.shippers: [Shipper] = shippers
        self.nodes: [Node] = nodes
        self.loads: [Load] = loads

        self.new_loads: [Load] = self.loads.copy()
        self._assign_new_loads()

        self.distances = {}

    def iteration(self):
        """This is the main function of the Environment class. It represents the operation of the game for one unit of
        time. It should be called in a loop after initializing the game."""

        # getting new loads and broadcasting
        self._get_new_loads()
        self._assign_new_loads()

        # running auctions
        for node in self.nodes:
            node.run_auction()  # TODO: make this function

        # one distance unit for the moving carriers
        for carrier in self.carriers:
            carrier.next_state()  # TODO: make this function

        # getting new info and broadcasting
        new_infos: [Info] = []
        for load in self.loads:
            new_infos = new_infos + load.new_info()
        for node in self.nodes:
            node.update_weights_with_new_info(new_infos)  # TODO: implement this function

    def _get_new_loads(self):
        self.new_loads = []
        for shipper in self.shippers:
            new_shipper_loads = shipper.generate_loads()  # TODO: implement this function
            if new_shipper_loads is not None:
                self.new_loads = self.new_loads + new_shipper_loads
        self.loads = self.loads + self.new_loads

    def _assign_new_loads(self):
        for load in self.new_loads:
            load.start.waiting_loads.append(load)  # TODO: make sure the name of the implementation match

# FIXME: test