"""
This is the learning driver for the game with learning agents.
As you may see, it is more a script than a class.

One may not that it is not a tf_agents.driver.Driver because this class is a bit inappropriate to what we do.
    We have nothing like an environment, policies are part of the LearningCarriers and observers are part of the
    LearningAgents

It is supposed to:
    * Instantiate everything
    * Run an iteration loop on the environment (During which the Carriers will share their experience
        in a replay buffer)
    * Regularly ask the learning agent to update
    * Regularly change the costs parameters of the Carriers
    * Save the model if we are satisfied
"""

from tf_agents.utils.common import function as tfa_function

from PI_RPS.Games.Learning_Game.initialize import load_env_and_agent

# e = TFAEnvironment()
#
# ps = DummyNode('Paris', {}, 100, [], e)
# bx = DummyNode('Brussels', {}, 100, [], e)
# hh = DummyNode('Hamburg', {}, 100, [], e)
#
# for node in e.Nodes:
#     node.initialize_weights()
# del node
#
# e.build_node_state()
#
# distances = {ps: {bx: 3, hh: 6}, bx: {ps: 3, hh: 4}, hh: {ps: 6, bx: 4}}
# e.set_distance(distances)
# del distances
#
# DummyShipper('Paris->Hamburg', [NodeLaw(ps, hh, lambda: 1, {})], [], [], e)
# DummyShipper('Hamburg->Paris', [NodeLaw(hh, ps, lambda: 1, {})], [], [], e)
#
# # Initializing the agents
# time_step_spec = TimeStep(step_type=TensorSpec(shape=(), dtype=dtype('int32'), name='step_type'),
#                           reward=TensorSpec(shape=(), dtype=dtype('float32'), name='reward'),
#                           discount=BoundedTensorSpec(shape=(), dtype=dtype('float32'), name='discount',
#                                                      minimum=0.0, maximum=1.0),
#                           observation=TensorSpec(shape=(len(e.Nodes) + LearningCarrier.cost_dimension(),),
#                                                  dtype=dtype('float32'), name='observation'))
#
# action_spec = BoundedTensorSpec(shape=(len(e.Nodes),), dtype=dtype('float32'), name='action',
#                                 minimum=[100 for k in range(len(e.Nodes))],
#                                 maximum=[20000 for k in range(len(e.Nodes))])
#
# policy_spec = PolicyStep(action=action_spec, state=(), info=())
# data_spec = Transition(time_step=time_step_spec, action_step=policy_spec, next_time_step=time_step_spec)
#
# buffer = TFUniformReplayBuffer(data_spec=data_spec,
#                                batch_size=1,  # the sample batch size is then different, but we add 1 by 1
#                                max_length=3000,
#                                dataset_drop_remainder=True)
#
# actor_network = ActorNetwork(input_tensor_spec=time_step_spec.observation,
#                              output_tensor_spec=action_spec,
#                              fc_layer_params=(64, 64),
#                              dropout_layer_params=None,
#                              activation_fn=tf.keras.activations.relu,
#                              kernel_initializer='glorot_uniform',
#                              last_kernel_initializer='glorot_uniform'
#                              )
#
# critic_network = CriticNetwork(input_tensor_spec=(time_step_spec.observation, action_spec),
#                                observation_fc_layer_params=None,
#                                action_fc_layer_params=None,
#                                joint_fc_layer_params=(64, 64),
#                                joint_dropout_layer_params=None,
#                                activation_fn=tf.nn.relu,
#                                kernel_initializer='glorot_uniform',
#                                last_kernel_initializer='glorot_uniform',
#                                name='Critic_network')
#
# actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# exploration_noise_std = 150  # big enough for exploration, it may even be reduced over time
# critic_network_2 = None
# target_actor_network = None
# target_critic_network = None
# target_critic_network_2 = None
# target_update_tau = 1.0
# target_update_period = 1
# actor_update_period = 3
# td_errors_loss_fn = None  # we  don't need any since already given by the algo (elementwise huber_loss)
# gamma = 1
# reward_scale_factor = 1
# target_policy_noise = 0.2  # will default to 0.2
# target_policy_noise_clip = 0.5  # will default to 0.5: this is the min max of the noise
# gradient_clipping = None  # we don't want to clip the gradients (min max values)
# debug_summaries = False
# summarize_grads_and_vars = False
# train_step_counter = None  # should be automatically initialized
# name = "TD3_Multi_Agents_Learner"
#
# agent = LearningAgent(replay_buffer=buffer,
#                       time_step_spec=time_step_spec,
#                       action_spec=action_spec,
#                       actor_network=actor_network,
#                       critic_network=critic_network,
#                       actor_optimizer=actor_optimizer,
#                       critic_optimizer=critic_optimizer,
#                       exploration_noise_std=exploration_noise_std,
#                       critic_network_2=critic_network_2,
#                       target_actor_network=target_actor_network,
#                       target_critic_network=target_critic_network,
#                       target_critic_network_2=target_critic_network_2,
#                       target_update_tau=target_update_tau,
#                       target_update_period=target_update_period,
#                       actor_update_period=actor_update_period,
#                       td_errors_loss_fn=td_errors_loss_fn,
#                       gamma=gamma,
#                       reward_scale_factor=reward_scale_factor,
#                       target_policy_noise=target_policy_noise,
#                       target_policy_noise_clip=target_policy_noise_clip,
#                       gradient_clipping=gradient_clipping,
#                       debug_summaries=debug_summaries,
#                       summarize_grads_and_vars=summarize_grads_and_vars,
#                       train_step_counter=train_step_counter,
#                       name=name)
#
# # cleaning memory to debug
# del time_step_spec, action_spec, policy_spec, data_spec, buffer, actor_network, critic_network
# del actor_optimizer, critic_optimizer, exploration_noise_std, critic_network_2, target_actor_network
# del target_critic_network, target_critic_network_2, target_update_tau, target_update_period
# del actor_update_period, td_errors_loss_fn, gamma, reward_scale_factor, target_policy_noise
# del target_policy_noise_clip, gradient_clipping, debug_summaries, summarize_grads_and_vars
# del train_step_counter, name
#
# # Initializing the LearningCarriers in Learning Mode
# discount = 0.9
# for k in range(15):
#     LearningCarrier(name='CParis_{}'.format(k),
#                     home=ps,
#                     in_transit=False,
#                     next_node=ps,
#                     time_to_go=0,
#                     load=None,
#                     environment=e,
#                     episode_expenses=[],
#                     episode_revenues=[],
#                     this_episode_expenses=[],
#                     this_episode_revenues=0,
#                     transit_cost=3,
#                     far_from_home_cost=1,
#                     time_not_at_home=0,
#                     time_step=None,  # to initialize
#                     policy_step=None,  # same
#                     is_learning=True,
#                     learning_agent=agent,
#                     discount=discount,
#                     discount_power=1,  # To check
#                     )
#     LearningCarrier(name='CBrussels_{}'.format(k),
#                     home=bx,
#                     in_transit=False,
#                     next_node=bx,
#                     time_to_go=0,
#                     load=None,
#                     environment=e,
#                     episode_expenses=[],
#                     episode_revenues=[],
#                     this_episode_expenses=[],
#                     this_episode_revenues=0,
#                     transit_cost=3,
#                     far_from_home_cost=1,
#                     time_not_at_home=0,
#                     time_step=None,  # check
#                     policy_step=None,  # check
#                     is_learning=True,
#                     learning_agent=agent,  # To fill
#                     discount=discount,
#                     discount_power=1,  # To check
#                     )
#     LearningCarrier(name='CHamburg_{}'.format(k),
#                     home=hh,
#                     in_transit=False,
#                     next_node=hh,
#                     time_to_go=0,
#                     load=None,
#                     environment=e,
#                     episode_expenses=[],
#                     episode_revenues=[],
#                     this_episode_expenses=[],
#                     this_episode_revenues=0,
#                     transit_cost=3,
#                     far_from_home_cost=1,
#                     time_not_at_home=0,
#                     time_step=None,  # check
#                     policy_step=None,  # check
#                     is_learning=True,
#                     learning_agent=agent,  # To fill
#                     discount=discount,
#                     discount_power=1,  # To check
#                     )
#
# del ps, bx, hh

e, learning_agent = load_env_and_agent(n_carriers=11*5,
                                       discount=0.95,
                                       exploration_noise=500,
                                       target_update_tau_p=0.1,
                                       target_update_period_p=2,
                                       actor_update_period_p=2,
                                       reward_scale_factor_p=1/500,
                                       target_policy_noise_p=30,
                                       target_policy_noise_clip_p=75,
                                       max_time_not_at_home=1)


train = tfa_function(learning_agent.train)
# Defining the training loop
for k in range(100):
    print(k)
    e.iteration()
    e.shuffle_new_transition_carriers()
    for carrier in e.new_transition_carriers:
        experience, _ = next(carrier.training_data_set_iter)
        learning_agent.train(experience=experience, weights=None)
    e.clear_new_transition_carriers()
    # collect experience
    # agent.train(experience=, weights=None)
print('end')