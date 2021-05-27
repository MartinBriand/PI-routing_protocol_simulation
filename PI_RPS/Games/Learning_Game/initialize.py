"""
This files defines a few functions to initialize the variables in notebooks.
"""

import csv
import os
from typing import List, Dict, Tuple, Union

import random
import numpy as np

from numpy import dtype
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.specs import TensorSpec, BoundedTensorSpec
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.trajectory import Transition

import tensorflow as tf

from PI_RPS.Mechanics.Actors.Carriers.learning_carrier import LearningCarrier, LearningAgent
from PI_RPS.Mechanics.Actors.Nodes.dummy_node import DummyNode
from PI_RPS.Mechanics.Actors.Nodes.node import Node
from PI_RPS.Mechanics.Actors.Shippers.dummy_shipper import DummyShipper
from PI_RPS.Mechanics.Actors.Shippers.shipper import Shipper, NodeLaw
from PI_RPS.Mechanics.Environment.tfa_environment import TFAEnvironment
from PI_RPS.Mechanics.Tools.load import Load


def load_env_and_agent(n_carriers: int,
                       action_min: float,
                       action_max: float,
                       discount: float,
                       shippers_reserve_price_per_distance: float,
                       shipper_default_reserve_price: float,
                       node_nb_info: int,
                       info_cost_max_factor_increase: float,
                       init_node_weights_distance_scaling_factor: float,
                       max_nb_infos_per_load: int,
                       tnah_divisor: float,
                       exploration_noise: float,
                       target_update_tau_p: float,
                       target_update_period_p: int,
                       actor_update_period_p: int,
                       reward_scale_factor_p: float,
                       target_policy_noise_p: float,
                       target_policy_noise_clip_p: float,
                       max_time_not_at_home: int,
                       actor_fc_layer_params: Tuple,
                       actor_dropout_layer_params: Union[float, None],
                       critic_observation_fc_layer_params: Union[Tuple, None],
                       critic_action_fc_layer_params: Union[Tuple, None],
                       critic_joint_fc_layer_params: Tuple,
                       critic_joint_dropout_layer_params: Union[float, None],
                       actor_learning_rate: float,
                       critic_learning_rate: float,
                       buffer_max_length: int,
                       replay_buffer_batch_size: int
                       ) -> Tuple[TFAEnvironment, LearningAgent]:
    path = os.path.abspath(os.path.dirname(__file__))
    lambdas: np.ndarray = _read_csv(os.path.join(path, 'data/city_traffic_lambda_table.csv'))
    attribution: np.ndarray = _read_csv(os.path.join(path, 'data/city_traffic_dest_attribution_table.csv'))
    distances: np.ndarray = _read_csv(os.path.join(path, 'data/city_distance_matrix_time_step.csv'))

    # check size
    lts = lambdas.shape
    ats = attribution.shape
    dts = distances.shape
    assert lts[0] == ats[0] - 1 == ats[1] - 1 == dts[0] - 1 == dts[1] - 1, \
        "lambdas shape: {}\nattribution shape: {}\ndistance shape: {}".format(lts, ats, dts)

    # check keys
    n = lts[0]
    assert (lambdas[:, 0] == attribution[1:, 0]).sum() == n, \
        "keys do not match:\n{}\n{}".format(lambdas[:, 0], attribution[1:, 0])
    assert (lambdas[:, 0] == attribution[0, 1:]).sum() == n, \
        "keys do not match:\n{}\n{}".format(lambdas[:, 0], attribution[0, 1:])
    assert (lambdas[:, 0] == distances[1:, 0]).sum() == n, \
        "keys do not match:\n{}\n{}".format(lambdas[:, 0], distances[1:, 0])
    assert (lambdas[:, 0] == distances[0, 1:]).sum() == n, \
        "keys do not match:\n{}\n{}".format(lambdas[:, 0], distances[0, 1:])

    # make dicts
    lambdas, attribution, distances = _to_dicts(lambdas[:, 0], lambdas, attribution, distances)

    # create env
    e = TFAEnvironment(nb_hours_per_time_unit=6.147508,  # 390 km at an average speed of 39.42 km/h)
                       t_c_mu=33.,
                       t_c_sigma=4.15,
                       ffh_c_mu=20.,
                       ffh_c_sigma=1.00,  # multiplication by nb_hours occurs in init
                       max_nb_infos_per_load=max_nb_infos_per_load,
                       tnah_divisor=tnah_divisor,
                       action_min=action_min,
                       action_max=action_max,
                       init_node_weights_distance_scaling_factor=init_node_weights_distance_scaling_factor,
                       max_time_not_at_home=max_time_not_at_home)

    # create Nodes
    for name in lambdas.keys():
        DummyNode(name=name,
                  weights={},
                  nb_info=node_nb_info,
                  info_cost_max_factor_increase=info_cost_max_factor_increase,
                  revenues=[],
                  environment=e)

    e.build_node_state()

    lambdas, attribution, distances = _to_node_keys(e, lambdas, attribution, distances)
    e.set_distances(distances)

    for node in e.nodes:
        node.initialize_weights()

    # create Shippers
    shipper = DummyShipper(name='Shipper_arrete_de_shipper',
                           laws=[],
                           expenses=[],
                           loads=[],
                           environment=e,
                           reserve_price_per_distance=shippers_reserve_price_per_distance,
                           default_reserve_price=shipper_default_reserve_price)

    # create laws
    generator = np.random.default_rng()

    def law(load_shipper: Shipper,
            environment: TFAEnvironment,
            start_node: Node,
            lamb: float,
            population: List[Node],
            weights: List[float]) -> None:
        nb_loads = generator.poisson(lamb)
        for k in range(nb_loads):
            arrival_node = random.choices(population=population, weights=weights)[0]
            Load(departure=start_node, arrival=arrival_node, shipper=load_shipper, environment=environment)

    for start in lambdas.keys():
        params = {'load_shipper': shipper,
                  'environment': e,
                  'start_node': start,
                  'lamb': lambdas[start],
                  'population': list(attribution[start].keys()),
                  'weights': list(attribution[start].values())}
        shipper.add_law(NodeLaw(owner=shipper, law=law, params=params))

    # create Carriers

    data_spec = _init_learning_agent(e=e,
                                     exploration_noise=exploration_noise,
                                     target_update_tau_p=target_update_tau_p,
                                     target_update_period_p=target_update_period_p,
                                     actor_update_period_p=actor_update_period_p,
                                     reward_scale_factor_p=reward_scale_factor_p,
                                     target_policy_noise_p=target_policy_noise_p,
                                     target_policy_noise_clip_p=target_policy_noise_clip_p,
                                     actor_fc_layer_params=actor_fc_layer_params,
                                     actor_dropout_layer_params=actor_dropout_layer_params,
                                     critic_observation_fc_layer_params=critic_observation_fc_layer_params,
                                     critic_action_fc_layer_params=critic_action_fc_layer_params,
                                     critic_joint_fc_layer_params=critic_joint_fc_layer_params,
                                     critic_joint_dropout_layer_params=critic_joint_dropout_layer_params,
                                     actor_learning_rate=actor_learning_rate,
                                     critic_learning_rate=critic_learning_rate)

    learning_agent = e.learning_agent

    _init_learning_carriers(data_spec=data_spec,
                            n_carriers=n_carriers,
                            environment=e,
                            learning_agent=learning_agent,
                            discount=discount,
                            buffer_max_length=buffer_max_length,
                            replay_buffer_batch_size=replay_buffer_batch_size)

    return e, learning_agent


def _read_csv(file_path: str) -> np.ndarray:
    """Return a List with all the values"""
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        final: List = []
        for line in reader:
            final.append(line)
    return np.array(final)


def _to_dicts(keys: np.ndarray,
              lambdas: np.ndarray,
              attribution: np.ndarray,
              distances: np.ndarray) -> (Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]):
    """Transform the lists to dictionaries (easier to use)"""
    new_lambdas, new_attribution, new_distances = {}, {}, {}
    for i, key_i in enumerate(keys):
        new_attribution[key_i] = {}
        new_distances[key_i] = {}
        new_lambdas[key_i] = float(lambdas[i, 1])
        for j, key_j in enumerate(keys):
            if i != j:
                new_attribution[key_i][key_j] = float(attribution[i + 1, j + 1])
                new_distances[key_i][key_j] = int(distances[i + 1, j + 1])
    return new_lambdas, new_attribution, new_distances


def _to_node_keys(e: TFAEnvironment,
                  lambdas: Dict[str, float],
                  attribution: Dict[str, Dict[str, float]],
                  distances: Dict[str, Dict[str, int]]) -> (Dict[Node, float],
                                                            Dict[Node, Dict[Node, float]],
                                                            Dict[Node, Dict[Node, int]]):
    node_name_dict = {node.name: node for node in e.nodes}
    new_lambdas = {node_name_dict[name]: lamb for name, lamb in lambdas.items()}
    new_attribution = {node_name_dict[name1]: {node_name_dict[name2]: att for name2, att in obj.items()}
                       for name1, obj in attribution.items()}
    new_distances = {node_name_dict[name1]: {node_name_dict[name2]: dist for name2, dist in obj.items()}
                     for name1, obj in distances.items()}

    return new_lambdas, new_attribution, new_distances


def _init_learning_agent(e: TFAEnvironment,
                         exploration_noise: float,
                         target_update_tau_p: float,
                         target_update_period_p: int,
                         actor_update_period_p: int,
                         reward_scale_factor_p: float,
                         target_policy_noise_p: float,
                         target_policy_noise_clip_p: float,
                         actor_fc_layer_params: Tuple,
                         actor_dropout_layer_params: Union[float, None],  # not sure for the float
                         critic_observation_fc_layer_params: Union[Tuple, None],
                         critic_action_fc_layer_params: Union[Tuple, None],
                         critic_joint_fc_layer_params: Tuple,
                         critic_joint_dropout_layer_params: Union[float, None],
                         actor_learning_rate: float,
                         critic_learning_rate: float
                         ):
    # Initializing the agents
    time_step_spec = TimeStep(step_type=TensorSpec(shape=(), dtype=dtype('int32'), name='step_type'),
                              reward=TensorSpec(shape=(), dtype=dtype('float32'), name='reward'),
                              discount=BoundedTensorSpec(shape=(), dtype=dtype('float32'), name='discount',
                                                         minimum=0.0, maximum=1.0),
                              observation=TensorSpec(shape=(2 * len(e.nodes) + LearningCarrier.cost_dimension(),),
                                                     dtype=dtype('float32'), name='observation'))

    action_spec = BoundedTensorSpec(shape=(len(e.nodes),), dtype=dtype('float32'), name='action',
                                    minimum=[0 for _ in range(len(e.nodes))],
                                    maximum=[1 for _ in range(len(e.nodes))])  # they are normalized in the game

    policy_spec = PolicyStep(action=action_spec, state=(), info=())
    data_spec = Transition(time_step=time_step_spec, action_step=policy_spec, next_time_step=time_step_spec)

    actor_network = ActorNetwork(input_tensor_spec=time_step_spec.observation,
                                 output_tensor_spec=action_spec,
                                 fc_layer_params=actor_fc_layer_params,
                                 dropout_layer_params=actor_dropout_layer_params,
                                 activation_fn=tf.keras.activations.relu,
                                 kernel_initializer='glorot_uniform',
                                 last_kernel_initializer='glorot_uniform'
                                 )

    critic_network = CriticNetwork(input_tensor_spec=(time_step_spec.observation, action_spec),
                                   observation_fc_layer_params=critic_observation_fc_layer_params,
                                   action_fc_layer_params=critic_action_fc_layer_params,
                                   joint_fc_layer_params=critic_joint_fc_layer_params,
                                   joint_dropout_layer_params=critic_joint_dropout_layer_params,
                                   activation_fn=tf.nn.relu,
                                   kernel_initializer='glorot_uniform',
                                   last_kernel_initializer='glorot_uniform',
                                   name='Critic_network')

    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)
    exploration_noise_std = exploration_noise / (e.action_max - e.action_min)  # big enough for exploration
    # it may even be reduced over time
    critic_network_2 = None
    target_actor_network = None
    target_critic_network = None
    target_critic_network_2 = None
    target_update_tau = target_update_tau_p  # default is 1 but books say better if small
    target_update_period = target_update_period_p  # 1 (default) might also be a good option
    actor_update_period = actor_update_period_p  # 1 (default) might also be a good option
    td_errors_loss_fn = None  # we  don't need any since already given by the algo (elementwise huber_loss)
    gamma = 1
    reward_scale_factor = reward_scale_factor_p
    target_policy_noise = target_policy_noise_p / (e.action_max - e.action_min)  # noise of the actions
    target_policy_noise_clip = target_policy_noise_clip_p / (
            e.action_max - e.action_min)  # will default to 0.5: this is the min max of the noise
    gradient_clipping = None  # we don't want to clip the gradients (min max values)
    debug_summaries = False
    summarize_grads_and_vars = False
    train_step_counter = None  # should be automatically initialized
    name = "TD3_Multi_Agents_Learner"

    learning_agent = LearningAgent(environment=e,
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

    learning_agent.initialize()

    return data_spec


def _init_learning_carriers(data_spec,
                            n_carriers: int,
                            environment: TFAEnvironment,
                            learning_agent: LearningAgent,
                            discount: float,
                            buffer_max_length: int,
                            replay_buffer_batch_size: int
                            ) -> None:
    counter = {}
    for k in range(n_carriers):
        node = environment.nodes[k % len(environment.nodes)]
        if node in counter.keys():
            counter[node] += 1
        else:
            counter[node] = 1
        road_costs = random.normalvariate(mu=environment.t_c_mu, sigma=environment.t_c_sigma)
        drivers_costs = random.normalvariate(mu=environment.ffh_c_mu, sigma=environment.ffh_c_sigma)
        buffer = TFUniformReplayBuffer(data_spec=data_spec,
                                       batch_size=1,  # the sample batch size is then different, but we add 1 by 1
                                       max_length=buffer_max_length,
                                       dataset_drop_remainder=True)
        LearningCarrier(name=node.name + '_' + str(counter[node]),
                        home=node,
                        in_transit=False,
                        next_node=node,
                        time_to_go=0,
                        load=None,
                        environment=environment,
                        episode_expenses=[],
                        episode_revenues=[],
                        this_episode_expenses=[],
                        this_episode_revenues=0,
                        transit_cost=road_costs,
                        far_from_home_cost=drivers_costs,
                        time_not_at_home=0,
                        learning_agent=learning_agent,
                        replay_buffer=buffer,
                        replay_buffer_batch_size=replay_buffer_batch_size,
                        is_learning=True,
                        discount=discount,
                        discount_power=1,
                        time_step=None,
                        policy_step=None)
        # note: admin_costs are defined in the CarrierWithCosts class
