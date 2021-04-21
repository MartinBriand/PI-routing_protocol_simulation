"""
Learning Carrier:
We define two objects:
    * A learning agent
    * The carriers

Their behavior will be different during learning and exploiting

During learning:
    * The agent
        * Learns according to the orders of the driver. It is the extension of a TD3 tf_agents.
        * The driver will feed the agent with data from a replay buffer and ask him to learn
        * The driver will answer requests from carriers to know what to do next
    * The carriers
        * Will ask agent what to bid
        * Will generate the transition to be stored in the replay buffer
        * Will have internal parameter being changed by the driver to help the agent learn with all possible parameters
            These parameters are just the parameters of the price functions

During exploitation
    * The agent will not learn anymore
    * The driver won't change the parameters of the carriers on the way
    * The carriers won't generate episodes or these episodes won't be added to a replay buffer
"""
# TODO: is this description correct at the end of the implementation?

from random import random
from tensorflow import constant as tf_constant, concat as tf_concat
from tf_agents.agents.td3.td3_agent import Td3Agent

from Mechanics.Actors.carriers.carrier import CarrierWithCosts

from typing import TYPE_CHECKING, Optional, List
from prjtyping.types import CarrierBid

if TYPE_CHECKING:
    from Mechanics.Actors.nodes.node import Node
    from Mechanics.Tools.load import Load
    from Mechanics.Environment.tfa_environment import TFAEnvironment


class LearningCarrier(CarrierWithCosts):  # , TFEnvironment):
    """

    """

    # TODO: add description here

    def __init__(self,
                 name: str,
                 home: 'Node',
                 in_transit: bool,
                 next_node: 'Node',
                 time_to_go: int,
                 load: Optional['Load'],
                 environment: 'TFAEnvironment',
                 episode_expenses: List[float],
                 episode_revenues: List[float],
                 this_episode_expenses: List[float],
                 this_episode_revenues: float,
                 transit_cost: float,
                 far_from_home_cost: float,
                 time_not_at_home: int,
                 learning_agent: 'LearningAgents',
                 is_learning: bool) -> None:  # TODO this function is useless for the moment

        super().__init__(name=name,
                         home=home,
                         in_transit=in_transit,
                         next_node=next_node,
                         time_to_go=time_to_go,
                         load=load,
                         environment=environment,
                         episode_expenses=episode_expenses,
                         episode_revenues=episode_revenues,
                         this_episode_expenses=this_episode_expenses,
                         this_episode_revenues=this_episode_revenues,
                         transit_cost=transit_cost,
                         far_from_home_cost=far_from_home_cost,
                         time_not_at_home=time_not_at_home)

        self._learning_agent = learning_agent
        self._is_learning = is_learning
        if self._is_learning:
            self._policy = self._learning_agent.collect_policy
        else:
            self._policy = self._learning_agent.policy

    # Carrier's methods
    def _decide_next_node(self) -> 'Node':  # Very simple function to get back home in 20% of the lost auctions
        # This could get smarter later
        """Decide of a next nodes after losing an auction (can be the same nodes when needed)"""
        home = random() < 0.2
        if home:
            return self._home
        else:
            return self._next_node

    def bid(self, node: 'Node') -> 'CarrierBid':
        #
        #
        # save state
        # ask the agents what to bid
        # save agent bid
        bid = {}
        # transform the agent bid in a CarrierBid format
        return bid

    def set_new_cost_parameters(self) -> None:
        pass

    def next_step(self) -> None:
        super().next_step()
        if self._is_learning and not self._in_transit:
            self._generate_episode()

    def _generate_episode(self) -> None:
        node_state = self._environment.this_node_state(self._next_node)
        cost_state = tf_constant([self._t_c, self._ffh_c, self._time_not_at_home])
        new_state = tf_concat([node_state, cost_state], 0)

        # generate episode
        # make sure the format of the state is the same as the one required by the agents
        self._old_tf_state = new_state

        pass

    @property
    def is_learning(self):
        return self._is_learning

    @is_learning.setter
    def is_learning(self, value: bool):
        self._is_learning = value
        if self._is_learning:
            self._policy = self._learning_agent.collect_policy
        else:
            self._policy = self._learning_agent.policy


class LearningAgents(Td3Agent):  # TODO: implement this
    pass
