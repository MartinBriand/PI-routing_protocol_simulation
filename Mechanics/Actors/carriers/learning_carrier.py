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
from tensorflow.python.framework.ops import EagerTensor
from tf_agents.agents.td3.td3_agent import Td3Agent
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep, StepType
from tf_agents.trajectories.trajectory import mid

from Mechanics.Actors.carriers.carrier import CarrierWithCosts

from typing import TYPE_CHECKING, Optional, List

from prj_typing.types import CarrierBid

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
                 is_learning: bool,
                 gamma: float,
                 time_step: Optional[TimeStep],
                 action: Optional[PolicyStep]) -> None:  # TODO this function is useless for the moment

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

        self._learning_agent: 'LearningAgents' = learning_agent

        if gamma > 1 or gamma < 0:
            raise ValueError('Gamma between 0 and 1')

        self._gamma: EagerTensor = tf_constant([gamma])
        self._gamma_power: int = 1

        self._is_learning: bool = is_learning
        if self._is_learning:
            self._policy = self._learning_agent.collect_policy
        else:
            self._policy = self._learning_agent.policy

        if time_step:
            self._time_step: Optional[TimeStep] = time_step
        else:
            self._time_step: Optional[TimeStep] = self._generate_current_time_step()

        if action:
            self._action_step: Optional[PolicyStep] = action
        else:
            self._action_step: Optional[PolicyStep] = None

    # Carrier's methods
    def _decide_next_node(self) -> 'Node':  # Very simple function to get back home in 20% of the lost auctions
        # This could get smarter later
        """Decide of a next nodes after losing an auction (can be the same nodes when needed)"""
        home = random() < 0.2
        if home:
            return self._home
        else:
            return self._next_node

    def get_attribution(self, load: 'Load', next_node: 'Node') -> None:
        super().get_attribution(load, next_node)
        self._gamma_power = self._time_to_go

    def dont_get_attribution(self) -> None:
        super().dont_get_attribution()
        if self._in_transit:
            self._gamma_power = self._time_to_go
        else:
            self._gamma_power = 1

    def bid(self, node: 'Node') -> 'CarrierBid':  # TODO
        self._action_step = self._policy.action(self._time_step)  # the time step is generated in next_step
        action = self._action_step.action.numpy()
        node_list = self._environment.nodes
        bid = {}
        for k in range(action.shape[0]):
            bid[node_list[k]] = action[k]
        return bid

    def set_new_cost_parameters(self, t_c: float, ffh_c: float) -> None:
        self._t_c = t_c
        self._ffh_c = ffh_c

    def next_step(self) -> None:
        super().next_step()
        if not self._in_transit and self._is_learning:
            new_time_step = self._generate_current_time_step()
            if self._is_learning:
                self._generate_trajectory(new_time_step)
            self._time_step = new_time_step

    def _generate_trajectory(self, new_time_step: TimeStep) -> None:  # TODO
        if self._action_step:
            trajectory = mid(observation=self._time_step.observation,
                             action=self._action_step.action,
                             policy_info=self._action_step.info,
                             reward=new_time_step.reward,
                             discount=self._gamma ** self._gamma_power)  # note that reward is update in next_step()
            # communicate trajectory
            # TODO: How do I know the next state in the trajectory?

    def _generate_current_time_step(self) -> TimeStep:
        node_state = self._environment.this_node_state(self._next_node)
        cost_state = tf_constant([self._t_c, self._ffh_c, self._time_not_at_home])

        # note that reward is update in next_step()
        # and that gamma_power is updated in the _get_attribution()/_dont_get_attribution()
        return TimeStep(step_type=StepType.MID,
                        reward=tf_constant([self._episode_revenues[-1] - self._episode_expenses[-1]]),
                        discount=self._gamma ** self._gamma_power,
                        osbervation=tf_concat([node_state, cost_state], 0))

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
