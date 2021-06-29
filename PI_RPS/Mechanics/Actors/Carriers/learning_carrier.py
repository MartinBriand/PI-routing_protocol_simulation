"""
Learning Carrier:
We define two objects:
    * A learning agent
    * The Carriers

Their behavior will be different during learning and exploiting

During learning:
    * The agent
        * Learns according to the orders of the driver. It is the extension of a TD3 tf_agents.
        * The driver will feed the agent with data from a replay buffer and ask him to learn
        * The driver will answer requests from Carriers to know what to do next
    * The Carriers
        * Will ask agent what to bid
        * Will generate the transition to be stored in the replay buffer
        * Will have internal parameter being changed by the driver to help the agent learn with all possible parameters
            These parameters are just the parameters of the price functions

During exploitation
    * The agent will not learn anymore
    * The driver won't change the parameters of the Carriers on the way
    * The Carriers won't generate episodes or these episodes won't be added to a replay buffer
"""
import abc

from tensorflow import constant as tf_constant, concat as tf_concat, expand_dims as tf_expand_dims
from tensorflow.python.data import Dataset
from tensorflow.python.framework.ops import EagerTensor
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep, StepType
from tf_agents.trajectories.trajectory import Transition

from PI_RPS.Mechanics.Actors.Carriers.carrier import CarrierWithCosts, MultiBidCarrier, SingleBidCarrier

from typing import TYPE_CHECKING, Optional, List, Tuple

from PI_RPS.Mechanics.Actors.Carriers.learning_agent import LearningAgent

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Actors.Nodes.node import Node
    from PI_RPS.Mechanics.Tools.load import Load
    from PI_RPS.Mechanics.Environment.tfa_environment import TFAEnvironment
    from PI_RPS.prj_typing.types import CarrierMultiBid, CarrierSingleBid


class LearningCarrier(CarrierWithCosts, abc.ABC):
    """
    Base abstract class for learning carrier
    It is a carrier but:
        * getting normalized action info from agents and then bidding according to that
        * set the discount power and generate time_steps and transitions
        * is able to change its parameters
        * go back home when not seeing your boss (or mother) for a too long time
    """

    def __init__(self,
                 name: str,
                 home: 'Node',
                 nb_lost_auctions_in_a_row: int,
                 max_lost_auctions_in_a_row: int,
                 in_transit: bool,
                 previous_node: 'Node',
                 next_node: 'Node',
                 time_to_go: int,
                 load: Optional['Load'],
                 environment: 'TFAEnvironment',
                 episode_types: List[Tuple[str, 'Node', 'Node']],
                 episode_expenses: List[float],
                 episode_revenues: List[float],
                 this_episode_expenses: List[float],
                 this_episode_revenues: float,
                 transit_cost: float,
                 far_from_home_cost: float,
                 time_not_at_home: int,
                 learning_agent: 'LearningAgent',
                 replay_buffer: ReplayBuffer,
                 replay_buffer_batch_size: int,
                 is_learning: bool,
                 discount: float,
                 discount_power: Optional[int],
                 time_step: Optional[TimeStep],
                 policy_step: Optional[PolicyStep]) -> None:

        super().__init__(name=name,
                         home=home,
                         in_transit=in_transit,
                         previous_node=previous_node,
                         next_node=next_node,
                         time_to_go=time_to_go,
                         load=load,
                         environment=environment,
                         episode_types=episode_types,
                         episode_expenses=episode_expenses,
                         episode_revenues=episode_revenues,
                         this_episode_expenses=this_episode_expenses,
                         this_episode_revenues=this_episode_revenues,
                         transit_cost=transit_cost,
                         far_from_home_cost=far_from_home_cost,
                         time_not_at_home=time_not_at_home)

        self._action_scale: float = self._environment.action_max - self._environment.action_min
        self._action_shift: float = self._environment.action_min

        self._nb_lost_auctions_in_a_row = nb_lost_auctions_in_a_row
        self._max_lost_auctions_in_a_row = max_lost_auctions_in_a_row

        self._t_c_obs = (self._t_c - self._environment.t_c_mu) / self._environment.t_c_sigma
        self._ffh_c_obs = (self._ffh_c - self._environment.ffh_c_mu) / self._environment.ffh_c_sigma

        self._learning_agent: 'LearningAgent' = learning_agent
        self._learning_agent.add_carrier(self)

        self._replay_buffer: ReplayBuffer = replay_buffer
        self._replay_buffer_batch_size: int = replay_buffer_batch_size
        self._training_data_set: Dataset = self._replay_buffer.as_dataset(
            sample_batch_size=self._replay_buffer_batch_size,
            num_steps=None,
            num_parallel_calls=None,
            single_deterministic_pass=False
        )
        self._training_data_set_iter = iter(self._training_data_set)

        assert 0 <= discount <= 1, 'Discount between 0 and 1'

        self._discount: EagerTensor = tf_constant([discount])
        self._discount_power: int = discount_power if discount_power else 1

        self._register_real_cost: bool = True  # when reserve price is involved in winning we do not want to register
        # the real cost in the transition but 0 instead

        self._is_learning: bool = is_learning
        if self._is_learning:
            self._policy: TFPolicy = self._learning_agent.collect_policy
        else:
            self._policy: TFPolicy = self._learning_agent.policy

        self._is_first_step: bool = (time_step is None)
        if not self._is_first_step:
            self._time_step: TimeStep = time_step
        self.init_first_step()

        self._policy_step: Optional[PolicyStep] = policy_step

    def _decide_next_node(self) -> 'Node':
        """
        Go home only if more than self._max_time_not_at_home since last time at home
        """
        if self._nb_lost_auctions_in_a_row > self._max_lost_auctions_in_a_row:
            return self._home
        else:
            return self._next_node

    def get_attribution(self, load: 'Load', next_node: 'Node', reserve_price_involved: bool) -> None:
        super().get_attribution(load, next_node, reserve_price_involved)
        self._nb_lost_auctions_in_a_row = 0
        self._discount_power = self._time_to_go
        self._register_real_cost = not reserve_price_involved

    def dont_get_attribution(self) -> None:
        self._nb_lost_auctions_in_a_row += 1
        super().dont_get_attribution()
        if self._in_transit:
            self._discount_power = self._time_to_go
        else:
            self._discount_power = 1

        self._register_real_cost = True

    def next_step(self) -> None:
        """
        This version of next step takes care of generating transitions and time steps.
        If not at home for a too long time, we go home and don't record the transition.
        """
        super().next_step()  # this takes care of writing the transitions to record data
        if not self._in_transit:
            next_time_step = self._generate_current_time_step()
            if self._is_learning and not self._is_first_step:
                self._generate_transition(next_time_step)
            self._time_step = next_time_step
            self._is_first_step = False

    def _generate_transition(self, next_time_step: TimeStep) -> None:
        transition = Transition(time_step=self._time_step,
                                action_step=self._policy_step,
                                next_time_step=next_time_step)

        # note that reward is update in next_step()
        # communicate trajectory
        self._replay_buffer.add_batch(transition)
        if self._replay_buffer.num_frames().numpy() >= self._replay_buffer_batch_size:
            # to avoid training on repeated transitions
            self._environment.add_carrier_to_enough_transitions(self)

    def _generate_current_time_step(self) -> TimeStep:
        node_state = self._environment.this_node_state(self._next_node)
        home_state = self._environment.this_node_state(self._home)
        cost_state = tf_constant([self._t_c_obs,
                                  self._ffh_c_obs,
                                  self._time_not_at_home / self._environment.tnah_divisor], dtype='float32')
        observation = tf_expand_dims(tf_concat([node_state, home_state, cost_state], 0), axis=0)
        discount = self._discount ** self._discount_power
        if self._is_first_step:
            step_type = tf_constant([StepType.FIRST])  # step type is just for consistency but useless in our algo
            reward = tf_constant([0.], dtype='float32')
        elif not self._register_real_cost:
            step_type = tf_constant([StepType.MID])
            reward = tf_constant([0.], dtype='float32')
        else:
            step_type = tf_constant([StepType.MID])
            reward = tf_constant([self._episode_revenues[-1] - self._episode_expenses[-1]], dtype='float32')

        # note that reward is update in next_step()
        # and that discount_power is updated in the _get_attribution()/_dont_get_attribution()
        time_step = TimeStep(step_type=step_type,
                             reward=reward,
                             discount=discount,
                             observation=observation)

        return time_step

    def init_first_step(self):
        if not self._in_transit:
            self._is_first_step = True
            self._time_step: Optional[TimeStep] = self._generate_current_time_step()
            self._is_first_step = False
        else:
            self._is_first_step = True

    def _set_new_cost_parameters(self, t_c: float, ffh_c: float) -> None:
        """
        Setting new parameters for learners and resetting buffers
        """
        super()._set_new_cost_parameters(t_c, ffh_c)
        self._t_c_obs = (self._t_c - self._environment.t_c_mu) / self._environment.t_c_sigma
        self._ffh_c_obs = (self._ffh_c - self._environment.ffh_c_mu) / self._environment.ffh_c_sigma
        self._replay_buffer.clear()
        self._discount_power = 1
        self.init_first_step()

    def update_collect_policy(self):
        assert self._is_learning, "update_collect_policy only if learning"
        self._policy = self._learning_agent.collect_policy

    def set_not_learning(self):
        assert self._is_learning, "Set only learning agents to non-learning"
        self._is_learning = False
        self._policy = self._learning_agent.policy

    def set_learning(self):
        assert not self._is_learning, "Set only non-learning agent to learning"
        self._is_learning = True
        self._policy = self._learning_agent.collect_policy

    @property
    def is_learning(self) -> bool:
        return self._is_learning

    @property
    def training_data_set_iter(self):
        return self._training_data_set_iter


class MultiLanesLearningCarrier(LearningCarrier, MultiBidCarrier):  # , TFEnvironment):
    """
    The carrier is able to bid on all lanes
    """

    def bid(self) -> 'CarrierMultiBid':
        self._policy_step = self._policy.action(self._time_step)  # the time step is generated in next_step
        action = self._policy_step.action.numpy()
        action = action * self._action_scale + self._action_shift  # This way we can get back to normalized
        # actions in the env without interfering with TFA
        node_list = self._environment.nodes
        bid = {}
        for k in range(action.shape[-1]):
            next_node = node_list[k]
            if next_node != self._next_node:
                bid[next_node] = action[0, k]  # 0 because of the first dimension
        return bid


class MultiLanesLearningCarrier2(LearningCarrier, MultiBidCarrier):
    """
    This carrier will bid around its price
    """

    def bid(self) -> 'CarrierMultiBid':
        self._policy_step = self._policy.action(self._time_step)
        action = self._policy_step.action.numpy()
        action = action * self._action_scale + self._action_shift
        bid = {}
        for next_node in self._environment.nodes:
            if next_node != self._next_node:
                bid[next_node] = self._calculate_costs(self._next_node, next_node) * action[0, 0]
        return bid

    def _calculate_costs(self, from_node: 'Node', to_node: 'Node') -> float:
        """Will be called by bid"""
        result = 0.
        for delta_t in range(self._environment.get_distance(from_node, to_node)):
            t = self._time_not_at_home + delta_t
            result += self._transit_costs() + self._far_from_home_costs(time_not_at_home=t)
        return result


class SingleLaneLearningCarrier(MultiLanesLearningCarrier, SingleBidCarrier):
    """
    The carrier can only bid on destination lane
    """

    def bid(self, next_node: 'Node') -> 'CarrierSingleBid':
        # I know that the format of the super call doesn't wait for next_node
        multi_lanes_bid = super().bid()
        return multi_lanes_bid[next_node]


class SingleLaneLearningCarrier2(MultiLanesLearningCarrier2, SingleBidCarrier):
    """Same"""

    def bid(self, next_node: 'Node') -> 'CarrierSingleBid':
        # I know that the format of the super call doesn't wait for next_node
        multi_lanes_bid = super().bid()
        return multi_lanes_bid[next_node]
