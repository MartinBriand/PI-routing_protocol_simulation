"""
This is the environment. It should be seen as a simple clock necessary for the functioning of the game, but not as a
real entity of the game. If a version ever have to be implemented or become more realistic, the Environment should be
deleted.
"""


class Environment:

    def __init__(self, carriers=None, shippers=None, nodes=None, loads=None):
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
        self.carriers = carriers
        self.shippers = shippers
        self.nodes = nodes
        self.loads = loads

        self.new_loads = self.loads.copy()
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
            node.run_auction()

        # one distance unit for the moving carriers
        for carrier in self.carriers:
            carrier.next_state()

        # getting new info and broadcasting
        new_infos = []
        for load in self.loads:
            new_infos.append(load.new_info())
        for node in self.nodes:
            node.update_weights_with_new_info(new_infos)

    def _get_new_loads(self):
        self.new_loads = []
        for shipper in self.shippers:
            new_shipper_load = shipper.generate_load()
            if new_shipper_load is not None:
                self.new_loads.append(new_shipper_load)
        self.loads = self.loads + self.new_loads

    def _assign_new_loads(self):
        for load in self.new_loads:
            load.start.waiting_loads.append(load)
