"""
This is the learning driver for the game with learning agents.
As you may see, it is more a script than a class.

One may not that it is not a tf_agents.driver.Driver because this class is a bit inappropriate to what we do.
    We have nothing like an environment, policies are part of the LearningCarriers and observers are part of the
    LearningAgents

It is supposed to:
    * Instantiate everything
    * Run an iteration loop on the environment (During which the carriers will share their experience
        in a replay buffer)
    * Regularly ask the learning agent to update
    * Regularly change the costs parameters of the carriers
    * Save the model if we are satisfied
"""
from numpy import dtype
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.specs import ArraySpec, BoundedArraySpec
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.trajectory import Transition

import tensorflow as tf

from Mechanics.Environment.tfa_environment import TFAEnvironment
from Mechanics.Actors.nodes.dummy_node import DummyNode
from Mechanics.Actors.carriers.learning_carrier import LearningCarrier, LearningAgent
from Mechanics.Actors.shippers.dummy_shipper import DummyShipper
from Mechanics.Actors.shippers.shipper import NodeLaw

e = TFAEnvironment()

ps = DummyNode('Paris', {}, 100, [], e)
bx = DummyNode('Brussels', {}, 100, [], e)
hh = DummyNode('Hamburg', {}, 100, [], e)

for node in e.nodes:
    node.initialize_weights()
del node

e.build_node_state()

distances = {ps: {bx: 3, hh: 6}, bx: {ps: 3, hh: 4}, hh: {ps: 6, bx: 4}}
e.set_distance(distances)
del distances

DummyShipper('Paris->Hamburg', [NodeLaw(ps, hh, lambda: 1, {})], [], [], e)
DummyShipper('Hamburg->Paris', [NodeLaw(hh, ps, lambda: 1, {})], [], [], e)

# Initializing the agents
time_step_spec = TimeStep(step_type=ArraySpec(shape=(), dtype=dtype('int32'), name='step_type'),
                          reward=ArraySpec(shape=(), dtype=dtype('float32'), name='reward'),
                          discount=BoundedArraySpec(shape=(), dtype=dtype('float32'), name='discount',
                                                    minimum=0.0, maximum=1.0),
                          observation=ArraySpec(shape=(len(e.nodes) + LearningCarrier.cost_dimension()),
                                                dtype=dtype('float32'), name='observation'))

action_spec = ArraySpec(shape=(len(e.nodes)), dtype=dtype('float32'), name='observation')
policy = PolicyStep(action=action_spec, state=(), info=())
data_spec = Transition(time_step=time_step_spec, action_step=policy, next_time_step=time_step_spec)

buffer = TFUniformReplayBuffer(data_spec=data_spec,
                               batch_size=1,  # the sample batch size is then different, but we add 1 by 1
                               max_length=3000,
                               dataset_drop_remainder=True)

actor_network =  # TODO find the correct network
critic_network = CriticNetwork(
        (time_step_spec.observation, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=(64, 64),
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
exploration_noise_std = 0.1  # Note that it will be normalize before
# TODO check that will accept action in more than one dimension
critic_network_2 = None  # TODO check the weights at creation
target_actor_network = None  # TODO check the weights at creation
target_critic_network = None  # TODO check
target_critic_network_2 = None  # TODO check
target_update_tau = 1.0
target_update_period = 1
actor_update_period = 3
td_errors_loss_fn = None  # we  don't need any since already given by the algo (elementwise huber_loss)
gamma = 1
reward_scale_factor =  # TO Define
target_policy_noise = 0.2  # will default to 0.2
target_policy_noise_clip = 0.5  # will default to 0.5
# TODO why do we need that ?
gradient_clipping = None  # TODO understand that
debug_summaries = False
summarize_grads_and_vars = False
train_step_counter = None  # should be automatically initialized
name = "TD3_Multi_Agents_Learner"

agent = LearningAgent(replay_buffer=buffer,
                      time_step_spec=time_step_spec,
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

# Initializing the LearningCarriers in Learning Mode
discount = 0.9
for k in range(10):
    LearningCarrier(name='CParis_{}'.format(k),
                    home=ps,
                    in_transit=False,
                    next_node=ps,
                    time_to_go=0,
                    load=None,
                    environment=e,
                    episode_expenses=[],
                    episode_revenues=[],
                    this_episode_expenses=[],
                    this_episode_revenues=0,
                    transit_cost=3,
                    far_from_home_cost=1,
                    time_not_at_home=0,
                    time_step=None,  # to initialize
                    policy_step=None,  # same
                    is_learning=True,
                    learning_agent=agent,
                    discount=discount,
                    discount_power=1,  # To check
                    )
    LearningCarrier(name='CBrussels_{}'.format(k),
                    home=bx,
                    in_transit=False,
                    next_node=bx,
                    time_to_go=0,
                    load=None,
                    environment=e,
                    episode_expenses=[],
                    episode_revenues=[],
                    this_episode_expenses=[],
                    this_episode_revenues=0,
                    transit_cost=3,
                    far_from_home_cost=1,
                    time_not_at_home=0,
                    time_step=None,  # check
                    policy_step=None,  # check
                    is_learning=True,
                    learning_agent=agent,  # To fill
                    discount=discount,
                    discount_power=1,  # To check
                    )
    LearningCarrier(name='CHamburg_{}'.format(k),
                    home=hh,
                    in_transit=False,
                    next_node=hh,
                    time_to_go=0,
                    load=None,
                    environment=e,
                    episode_expenses=[],
                    episode_revenues=[],
                    this_episode_expenses=[],
                    this_episode_revenues=0,
                    transit_cost=3,
                    far_from_home_cost=1,
                    time_not_at_home=0,
                    time_step=None,  # check
                    policy_step=None,  # check
                    is_learning=True,
                    learning_agent=agent,  # To fill
                    discount=discount,
                    discount_power=1,  # To check
                    )

del ps, bx, hh

# Defining the training loop
# for k in range(10000):
#     e.iteration()
# print('end')
