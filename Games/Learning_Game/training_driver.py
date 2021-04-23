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

from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

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

discount = 0.9

# Initializing the agents
buffer = TFUniformReplayBuffer(data_spec=, batch_size=, max_length=, dataset_drop_remainder=True)
time_step_spec =
action_spec =
actor_network =
critic_network =
actor_optimizer =
critic_optimizer =
exploration_noise_std =
critic_network_2 =
target_actor_network =
target_critic_network =
target_critic_network_2 =
target_update_tau =
target_update_period =
actor_update_period =
td_errors_loss_fn =
gamma =
reward_scale_factor =
target_policy_noise =
target_policy_noise_clip =
gradient_clipping =
debug_summaries =
summarize_grads_and_vars =
train_step_counter =
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
