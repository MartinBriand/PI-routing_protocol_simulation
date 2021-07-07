"""
File for the LearningAgent class
"""

from tf_agents.agents.td3.td3_agent import Td3Agent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.networks import Network
from tf_agents.policies import actor_policy, gaussian_policy
from tf_agents.trajectories.time_step import TimeStep
import tf_agents.typing.types as tfa_types
from tf_agents.typing import types

from tensorflow import Variable

from typing import Optional, List, TYPE_CHECKING, Any

from PI_RPS.prj_typing.types import AllLearningCarrier

if TYPE_CHECKING:
    from PI_RPS.Mechanics.Environment.environment import Environment


class LearningAgent(Td3Agent):
    """
    This is an extension of the TD3Agent with
        * the ability to change its exploration noise over time
    """

    def _loss(self, experience: types.NestedTensor, weights: types.Tensor) -> Optional[LossInfo]:
        raise NotImplementedError

    def __init__(self,
                 environment: 'Environment',
                 time_step_spec: TimeStep,
                 action_spec: tfa_types.NestedTensor,
                 actor_network: Network,
                 critic_network: Network,
                 actor_optimizer: tfa_types.Optimizer,
                 critic_optimizer: tfa_types.Optimizer,
                 exploration_noise_std: tfa_types.Float = 0.1,
                 critic_network_2: Optional[Network] = None,
                 target_actor_network: Optional[Network] = None,
                 target_critic_network: Optional[Network] = None,
                 target_critic_network_2: Optional[Network] = None,
                 target_update_tau: tfa_types.Float = 1.0,
                 target_update_period: tfa_types.Int = 1,
                 actor_update_period: tfa_types.Int = 1,
                 td_errors_loss_fn: Optional[tfa_types.LossFn] = None,
                 gamma: tfa_types.Float = 1.0,
                 reward_scale_factor: tfa_types.Float = 1.0,
                 target_policy_noise: tfa_types.Float = 0.2,
                 target_policy_noise_clip: tfa_types.Float = 0.5,
                 gradient_clipping: Optional[tfa_types.Float] = None,
                 debug_summaries: bool = False,
                 summarize_grads_and_vars: bool = False,
                 train_step_counter: Optional[Variable] = None,
                 name: str = None,
                 key: Any = None) -> None:
        super().__init__(time_step_spec=time_step_spec,
                         action_spec=action_spec,
                         actor_network=actor_network,
                         critic_network=critic_network,
                         actor_optimizer=actor_optimizer,
                         critic_optimizer=critic_optimizer,
                         exploration_noise_std=exploration_noise_std,
                         critic_network_2=critic_network_2,
                         target_actor_network=target_actor_network,
                         target_critic_network=target_critic_network,
                         target_critic_network_2=target_critic_network_2,
                         target_update_tau=target_update_tau,
                         target_update_period=target_update_period,
                         actor_update_period=actor_update_period,
                         td_errors_loss_fn=td_errors_loss_fn,
                         gamma=gamma,
                         reward_scale_factor=reward_scale_factor,
                         target_policy_noise=target_policy_noise,
                         target_policy_noise_clip=target_policy_noise_clip,
                         gradient_clipping=gradient_clipping,
                         debug_summaries=debug_summaries,
                         summarize_grads_and_vars=summarize_grads_and_vars,
                         train_step_counter=train_step_counter,
                         name=name)

        self._environment = environment

        self._key = key

        self._base_policy = actor_policy.ActorPolicy(
            time_step_spec=time_step_spec, action_spec=action_spec,
            actor_network=self._actor_network, clip=False)

        self._environment.register_learning_agent(self, self._key)
        self._carriers: List['AllLearningCarrier'] = []
        # This list is going to be a copy of the carriers of the tfaenvironment, but i keep this for two reasons:
        #   * we may want in future version to make them different (learning from different structures)
        #   * It is better to make a reference to the learning agent when changing properties linked to the learning
        #       process (costs, is_learning) but it makes more sense to access them via environment
        #       when environment related (home...)

    def add_carrier(self, carrier: 'AllLearningCarrier'):
        """To be called by the learning agent at creation to signal its presence"""
        self._carriers.append(carrier)

    def change_exploration_noise_std(self, value: float) -> None:
        """
        This is to change the collect policy for all learning agents
        """
        # change collect policy of the learner
        self._exploration_noise_std = value / (self._environment.action_max - self._environment.action_min)
        self._collect_policy = gaussian_policy.GaussianPolicy(self._base_policy,
                                                              scale=self._exploration_noise_std,
                                                              clip=True)

        for carrier in self._carriers:
            if carrier.is_learning:
                carrier.update_collect_policy()

    def set_carriers_to_learning(self):
        for carrier in self._carriers:
            carrier.set_learning()

    def set_carriers_to_not_learning(self):
        for carrier in self._carriers:
            carrier.set_not_learning()

    @property
    def carriers(self) -> List['AllLearningCarrier']:
        return self._carriers

    @property
    def key(self):
        return self._key
