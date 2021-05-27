# -*- coding: utf-8 -*-
"""Copy of training.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zCuibgDlAF1xrU_HQRs9I-ZPzibdC-tX

# Preparation
Before executing anything, if you are on google colab using google computational power, and want to use GPU acceleration: click on `Runtime`, then on `Change runtime type`, select `GPU` as the `Hardware Accelerator`, and click `Save`.

# Installation and import
"""

#! pip install https://github.com/MartinBriand/PI-routing_protocol_simulation/archive/develop.tar.gz

from tf_agents.utils.common import function as tfa_function
from PI_RPS.Games.Learning_Game.initialize import load_env_and_agent
import numpy as np

"""# Initialization"""

n_carriers_per_node = 15 # @param {type:"integer"}
action_min = 100. # @param {type:"number"}
action_max = 20000. # @param {type:"number"}
discount = 0.95 # @param {type:"number"}

shippers_reserve_price = 20000. # @param{type:"number"}
init_node_weights_distance_scaling_factor = 1500. # @param{type:"number"}
node_nb_info = 100 # @param{type:"integer"}
info_cost_max_factor_increase = 1.3 # @param{type:"integer"}
max_nb_infos_per_load = 5 # @param{type:"integer"}

max_time_not_at_home = 30 # @param {type:"integer"}
tnah_divisor = 30. # keep at 30, not a parameter
reward_scale_factor_p = 1. / 500. # keep at 1./500., not a parameter

replay_buffer_batch_size = 5 # @param {type:"integer"}
buffer_max_length = 25 # @param{type:"integer"}

starting_exploration_noise = 500. # @param {type:"number"}
final_exploration_noise = 20. # @param {type:"number"}
exploration_noise = starting_exploration_noise # not a param

actor_fc_layer_params = (64, 64) # @param
actor_dropout_layer_params = None # @param
critic_observation_fc_layer_params = None # @param
critic_action_fc_layer_params = None # @param
critic_joint_fc_layer_params = (64, 64) # @param
critic_joint_dropout_layer_params = None # @param

target_update_tau_p = 0.1 # @param {type:"number"}
target_update_period_p = 2 # @param {type:"integer"}
actor_update_period_p = 2 # @param {type:"integer"}
actor_learning_rate = 0.001 # @param{type:"number"}
critic_learning_rate = 0.001 # @param{type:"number"}

target_policy_noise_p = 30. # @param {type:"number"}
target_policy_noise_clip_p = 75. # @param {type:"number"}

e, learning_agent = load_env_and_agent(n_carriers=11*n_carriers_per_node, # 11 is the number of nodes
                                       shippers_reserve_price=shippers_reserve_price,
                                       init_node_weights_distance_scaling_factor=init_node_weights_distance_scaling_factor,
                                       node_nb_info=node_nb_info,
                                       info_cost_max_factor_increase=info_cost_max_factor_increase,
                                       max_nb_infos_per_load=max_nb_infos_per_load,
                                       discount=discount,
                                       exploration_noise=exploration_noise,
                                       target_update_tau_p=target_update_tau_p,
                                       target_update_period_p=target_update_period_p,
                                       actor_update_period_p=actor_update_period_p,
                                       reward_scale_factor_p=reward_scale_factor_p,
                                       target_policy_noise_p=target_policy_noise_p,
                                       target_policy_noise_clip_p=target_policy_noise_clip_p,
                                       max_time_not_at_home=max_time_not_at_home,
                                       action_min=action_min,
                                       action_max=action_max,
                                       tnah_divisor=tnah_divisor,
                                       replay_buffer_batch_size=replay_buffer_batch_size,
                                       buffer_max_length=buffer_max_length,
                                       actor_learning_rate=actor_learning_rate,
                                       critic_learning_rate=critic_learning_rate,
                                       actor_fc_layer_params=actor_fc_layer_params,
                                       actor_dropout_layer_params=actor_dropout_layer_params,
                                       critic_observation_fc_layer_params=critic_observation_fc_layer_params,
                                       critic_action_fc_layer_params=critic_action_fc_layer_params,
                                       critic_joint_fc_layer_params=critic_joint_fc_layer_params,
                                       critic_joint_dropout_layer_params=critic_joint_dropout_layer_params
                                       )

train = tfa_function(learning_agent.train)

"""# Training loop
## Results structure
"""

all_results = {'carriers_profit': {'min': [],
                                   'quartile1': [],
                                   'quartile2': [],
                                   'quartile3': [],
                                   'max': [],
                                   'mean': []},
               'nb_loads': [],
               'nb_arrived_loads': [],
               'delivery_costs': {'min': [],
                                  'quartile1': [],
                                  'quartile2': [],
                                  'quartile3': [],
                                  'max': [],
                                  'mean': []},
               'nb_hops': {'min': [],
                           'quartile1': [],
                           'quartile2': [],
                           'quartile3': [],
                           'max': [],
                           'mean': []},
               'delivery_times': {'min': [],
                                  'quartile1': [],
                                  'quartile2': [],
                                  'quartile3': [],
                                  'max': [],
                                  'mean': []},

               }

"""## Evaluation"""

def clear_env(start: bool) -> None:
    e.clear_node_auctions()
    e.clear_loads()
    e.clear_carrier_profits()
    e.clear_shipper_expenses()
    if start:
        learning_agent.set_carriers_to_not_learning()
    else:
        learning_agent.set_carriers_to_learning()
    e.check_carriers_first_steps()


def test(num_iter_per_test):
    # clear
    clear_env(start=True)

    # Running environment
    for counter in range(num_iter_per_test):
        e.iteration()

    # Getting data
    carriers_profit = []
    for carrier_p in e.carriers:
        if len(carrier_p.episode_revenues) > 1:
            carriers_profit.append(sum(carrier_p.episode_revenues[1:]) + sum(carrier_p.episode_expenses[1:]))
        else:
            carriers_profit.append(0.)
    carriers_profit = np.array(carriers_profit)

    nb_loads = len(e.loads)
    nb_arrived_loads = 0
    total_delivery_costs = []
    nb_hops = []
    delivery_times = []
    for load_p in e.loads:
        if load_p.is_arrived:
            nb_arrived_loads += 1
            total_delivery_costs.append(load_p.total_delivery_cost())
            nb_hops.append(load_p.nb_hops())
            delivery_times.append(load_p.delivery_time())
    delivery_costs = np.array(total_delivery_costs)
    nb_hops = np.array(nb_hops)
    delivery_times = np.array(delivery_times)

    results = {'carriers_profit': {'min': np.min(carriers_profit),
                                   'quartile1': np.quantile(carriers_profit, 0.25),
                                   'quartile2': np.quantile(carriers_profit, 0.5),
                                   'quartile3': np.quantile(carriers_profit, 0.75),
                                   'max': np.max(carriers_profit),
                                   'mean': np.mean(carriers_profit)},
               'nb_loads': nb_loads,
               'nb_arrived_loads': nb_arrived_loads,
               'delivery_costs': {'min': np.min(delivery_costs),
                                  'quartile1': np.quantile(delivery_costs, 0.25),
                                  'quartile2': np.quantile(delivery_costs, 0.5),
                                  'quartile3': np.quantile(delivery_costs, 0.75),
                                  'max': np.max(delivery_costs),
                                  'mean': np.mean(delivery_costs)},
               'nb_hops': {'min': np.min(nb_hops),
                           'quartile1': np.quantile(nb_hops, 0.25),
                           'quartile2': np.quantile(nb_hops, 0.5),
                           'quartile3': np.quantile(nb_hops, 0.75),
                           'max': np.max(nb_hops),
                           'mean': np.mean(nb_hops)},
               'delivery_times': {'min': np.min(delivery_times),
                                  'quartile1': np.quantile(delivery_times, 0.25),
                                  'quartile2': np.quantile(delivery_times, 0.5),
                                  'quartile3': np.quantile(delivery_times, 0.75),
                                  'max': np.max(delivery_times),
                                  'mean': np.mean(delivery_times)}

               }

    # clear
    clear_env(start=False)
    return results

keys_with_stats = ['carriers_profit', 'delivery_costs', 'nb_hops', 'delivery_times']
keys_without_stat = ['nb_loads', 'nb_arrived_loads']
stat_keys = ['min', 'quartile1', 'quartile2', 'quartile3', 'max', 'mean']


def add_results(results) -> None:
    for key_with_stats in keys_with_stats:
        for stat_key in stat_keys:
            all_results[key_with_stats][stat_key].append(results[key_with_stats][stat_key])
    for key_without_stat in keys_without_stat:
        all_results[key_without_stat].append(results[key_without_stat])

"""## Loop"""

num_rounds = 25 # @param {type:"integer"}
num_cost_pass = 1 # @param {type:"integer"}
num_train_per_pass = 10 # @param {type:"integer"}
num_iteration_per_test = 10 # @param{type:"integer"}

exploration_noise_update = (starting_exploration_noise - final_exploration_noise) / (num_rounds - 1)

def change_costs():
    for carrier_p in learning_agent.carriers:
        carrier_p.random_new_cost_parameters()

# add an ETA
for i in range(num_rounds):
    print("Test", i+1, '/', num_rounds)
    change_costs()
    print(e.nodes[0].readable_weights())
    test_results = test(num_iteration_per_test)
    print(test_results)
    add_results(test_results)
    for j in range(num_cost_pass):
        # print("Pass", j+1, "/", num_cost_pass)
        change_costs()
        for k in range(num_train_per_pass):
            # print("Training", k+1, "/", num_train_per_pass)
            e.iteration()
            e.shuffle_enough_transitions_carriers()
            n = len(e.enough_transitions_carriers)
            for _ in range(n):
                carrier = e.pop_enough_transitions_carriers()
                experience, _ = next(carrier.training_data_set_iter)
                train(experience=experience, weights=None)
    exploration_noise -= exploration_noise_update
    learning_agent.change_exploration_noise_std(exploration_noise)

print("Final test")
change_costs()
print(e.nodes[0].readable_weights())
test_results = test(num_iteration_per_test)
print(test_results)
add_results(test_results)



print(read_weights(e.nodes[0]))
print(read_weights(e.nodes[1]))

