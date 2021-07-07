"""
this is a carrier learning from whole episode, it is developed as we have difficulties to obtain convergence with
the transition learning carriers. We hope that this would better converge.
"""

import abc

from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep, StepType
from tf_agents.trajectories.trajectory import Transition

from tensorflow import constant as tf_constant, expand_dims as tf_expand_dims
from tensorflow.python.data import Dataset

from typing import Optional, List, Tuple, TYPE_CHECKING

from PI_RPS.Mechanics.Actors.Carriers.carrier import CarrierWithCosts, MultiBidCarrier, SingleBidCarrier
from PI_RPS.Mechanics.Actors.Carriers.learning_agent import LearningAgent
from PI_RPS.prj_typing.types import CarrierMultiBid, CarrierSingleBid

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Tools.load import Load
    from PI_RPS.Mechanics.Environment.environment import Environment


class EpisodeLearningCarrier(CarrierWithCosts, abc.ABC):
    """
    This is a carrier learning from whole episodes. The idea is that they bid their costs, they have a parameter about
    which deviation from their costs they do and another one about how often they go home.
    The learning agent from which they set the deviation parameter belongs to their home.
    """

    def __init__(self,
                 name: str,
                 home: 'Node',
                 in_transit: bool,
                 previous_node: 'Node',
                 next_node: 'Node',
                 time_to_go: int,
                 load: Optional['Load'],
                 environment: 'Environment',
                 episode_types: List[Tuple[str, 'Node', 'Node', bool]],
                 episode_expenses: List[float],
                 episode_revenues: List[float],
                 this_episode_expenses: List[float],
                 this_episode_revenues: float,
                 transit_cost: Optional[float],
                 far_from_home_cost: Optional[float],
                 time_not_at_home: int,
                 nb_lost_auctions_in_a_row: int,
                 max_lost_auctions_in_a_row: int,
                 is_learning: bool,
                 time_step: Optional[TimeStep],
                 policy_step: Optional[PolicyStep],
                 replay_buffer: ReplayBuffer,
                 replay_buffer_batch_size: int,
                 episode_learning_agent: 'LearningAgent') -> None:

        super().__init__(name,
                         home,
                         in_transit,
                         previous_node,
                         next_node,
                         time_to_go,
                         load,
                         environment,
                         episode_types,
                         episode_expenses,
                         episode_revenues,
                         this_episode_expenses,
                         this_episode_revenues,
                         transit_cost,
                         far_from_home_cost,
                         time_not_at_home)

        self._action_scale: float = self._environment.action_max - self._environment.action_min
        self._action_shift: float = self._environment.action_min

        self._nb_lost_auctions_in_a_row: int = nb_lost_auctions_in_a_row
        self._max_lost_auctions_in_a_row: int = max_lost_auctions_in_a_row

        self._t_c_obs = (self._t_c - self._environment.t_c_mu) / self._environment.t_c_sigma
        self._ffh_c_obs = (self._ffh_c - self._environment.ffh_c_mu) / self._environment.ffh_c_sigma

        self._reserve_price_involved_in_episode = False

        self._episode_learning_agent = episode_learning_agent
        self._episode_learning_agent.add_carrier(self)

        self._replay_buffer: ReplayBuffer = replay_buffer
        self._replay_buffer_batch_size: int = replay_buffer_batch_size
        self._training_data_set: Dataset = self._replay_buffer.as_dataset(
            sample_batch_size=self._replay_buffer_batch_size,
            num_steps=None,
            num_parallel_calls=None,
            single_deterministic_pass=False
        )
        self._training_data_set_iter = iter(self._training_data_set)

        self._is_learning: bool = is_learning
        if self._is_learning:
            self._policy: TFPolicy = self._episode_learning_agent.collect_policy
        else:
            self._policy: TFPolicy = self._episode_learning_agent.policy

        self._is_first_step = (time_step is None)
        if not self._is_first_step:
            self._initial_time_step: TimeStep = time_step
        else:
            assert policy_step is None, "policy_step and time_step should both be none"
            self._init_first_step()

        self._policy_step: PolicyStep = self._policy.action(self._initial_time_step) if policy_step is None \
            else policy_step

        self._cost_majoration: float = self._policy_step.action.numpy()[0, 0] * self._action_scale + self._action_shift

    def _decide_next_node(self) -> 'Node':
        """
        Go home only if more than self._max_time_not_at_home since last time at home
        """
        if self._nb_lost_auctions_in_a_row > self._max_lost_auctions_in_a_row:
            return self._home
        else:
            return self._next_node

    def dont_get_attribution(self) -> None:
        self._nb_lost_auctions_in_a_row += 1
        super().dont_get_attribution()

    def get_attribution(self, load: 'Load', next_node: 'Node', reserve_price_involved: bool) -> None:
        super().get_attribution(load, next_node, reserve_price_involved)
        self._nb_lost_auctions_in_a_row = 0
        self._reserve_price_involved_in_episode = self._reserve_price_involved_in_episode or reserve_price_involved

    def _calculate_costs(self, from_node: 'Node', to_node: 'Node') -> float:
        """Will be called by bid"""
        result = 0.
        for delta_t in range(self._environment.get_distance(from_node, to_node)):
            t = self._time_not_at_home + delta_t
            result += self._transit_costs() + self._far_from_home_costs(time_not_at_home=t)
        return result

    def next_step(self) -> None:
        """
        This version of next step generates an initial time step when needed.
        """
        super().next_step()
        if self._is_first_step:
            self._init_first_step()

    def _init_first_step(self):
        assert self._is_first_step, "only call if we have a first step"
        if not self._in_transit:
            self._set_initial_time_step()
            self._reserve_price_involved_in_episode = False

    def _set_initial_time_step(self):
        observation = tf_expand_dims(tf_constant([self._t_c_obs,
                                                  self._ffh_c_obs], dtype='float32'),
                                     axis=0)
        discount = tf_constant([1.], dtype='float32')
        step_type = tf_constant([StepType.FIRST])  # step type is just for consistency but useless in our algo
        reward = tf_constant([0.], dtype='float32')
        time_step = TimeStep(step_type=step_type,
                             reward=reward,
                             discount=discount,
                             observation=observation)

        self._is_first_step = False
        self._initial_time_step = time_step
        self._policy_step = self._policy.action(self._initial_time_step)
        self._cost_majoration: float = self._policy_step.action.numpy()[0, 0] * self._action_scale + self._action_shift

    def finish_this_step_and_prepare_next_step(self) -> None:
        """
        Here we generate the transition and prepare for the next iteration.
        We do not change the costs.
        We will only change the costs after the learning.
        The learning occurs only if all buffers are full.
        """
        # generate new time step
        observation = tf_expand_dims(tf_constant([self._t_c_obs,
                                                  self._ffh_c_obs], dtype='float32'),
                                     axis=0)
        discount = tf_constant([1.], dtype='float32')
        step_type = tf_constant([StepType.LAST])  # step type is just for consistency but useless in our algo
        reward = tf_constant([sum(self._episode_revenues) - sum(self._episode_expenses)], dtype='float32')

        next_time_step = TimeStep(step_type=step_type,
                                  reward=reward,
                                  discount=discount,
                                  observation=observation)

        # generate transition
        transition = Transition(time_step=self._initial_time_step,
                                action_step=self._policy_step,
                                next_time_step=next_time_step)

        if not self._reserve_price_involved_in_episode:
            # even if the buffer is full we write the new events (this changes nothing)
            self._replay_buffer.add_batch(transition)

        # prepare the next time first time step (do not change costs)
        self._is_first_step = True
        self._init_first_step()

    def _set_new_cost_parameters(self, t_c: float, ffh_c: float) -> None:
        """
        Setting new parameters for learners and resetting buffers
        """
        super()._set_new_cost_parameters(t_c, ffh_c)
        self._t_c_obs = (self._t_c - self._environment.t_c_mu) / self._environment.t_c_sigma
        self._ffh_c_obs = (self._ffh_c - self._environment.ffh_c_mu) / self._environment.ffh_c_sigma
        self._replay_buffer.clear()
        self._is_first_step = True
        self._init_first_step()

    def update_collect_policy(self):
        assert self._is_learning, "update_collect_policy only if learning"
        self._policy = self._episode_learning_agent.collect_policy

    def set_not_learning(self):
        assert self._is_learning, "Set only learning agents to non-learning"
        self._is_learning = False
        self._policy = self._episode_learning_agent.policy

    def set_learning(self):
        assert not self._is_learning, "Set only non-learning agent to learning"
        self._is_learning = True
        self._policy = self._episode_learning_agent.collect_policy

    @property
    def is_learning(self) -> bool:
        return self._is_learning

    @property
    def training_data_set_iter(self):
        return self._training_data_set_iter

    @property
    def cost_majoration(self):
        return self._cost_majoration

    @property
    def replay_buffer_is_full(self) -> bool:
        return self._replay_buffer.num_frames().numpy() >= self._replay_buffer.capacity


class MultiLanesEpisodeLearningCarrier(EpisodeLearningCarrier, MultiBidCarrier):
    """
    This carrier bids on multiple lanes
    """

    def bid(self) -> 'CarrierMultiBid':
        bid = {}
        for next_node in self._environment.nodes:
            if next_node != self._next_node:
                bid[next_node] = self._calculate_costs(self._next_node, next_node) * self._cost_majoration
        return bid


class SingleLaneEpisodeLearningCarrier(EpisodeLearningCarrier, SingleBidCarrier):
    """
    This carrier bids on a single lane: the one which is the destination of the load
    """

    def bid(self, next_node: 'Node') -> 'CarrierSingleBid':
        """The bid function"""
        return self._calculate_costs(self._next_node, next_node) * self._cost_majoration
