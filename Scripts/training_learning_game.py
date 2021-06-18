# -*- coding: utf-8 -*-
"""
Training script for the learning carriers (still with bugs for the moment)
"""

from tf_agents.utils.common import function as tfa_function
from PI_RPS.Games.Learning_Game.initialize import load_tfa_env_and_agent
import numpy as np
import time

"""# Initialization"""

n_carriers_per_node = 30  # @param {type:"integer"}
action_min = 0.  # @param {type:"number"}
action_max = 20000.  # @param {type:"number"}
discount = 0.95  # @param {type:"number"}

shippers_reserve_price_per_distance = 1200.  # @param{type:"number"}
shipper_default_reserve_price = 20000.  # @param{type:"number"}

init_node_weights_distance_scaling_factor = 500.  # @param{type:"number"}
max_node_weights_distance_scaling_factor = 500. * 1.3  # @param{type:"number"}
# should be big enough to be unrealistic.
# They won't be used if not learning nodes

node_auction_cost = 0.  # @param{type:"number"}
node_nb_info = 100  # @param{type:"integer"}
# not used if node is not learning

max_nb_infos_per_load = 5  # @param{type:"integer"}
# not used if not learning nodes

max_time_not_at_home = 30  # @param {type:"integer"}

tnah_divisor = 30.  # keep at 30, not a parameter
reward_scale_factor_p = 1. / 500.  # keep at 1./500., not a parameter

replay_buffer_batch_size = 5  # @param {type:"integer"}
buffer_max_length = 25  # @param{type:"integer"}

starting_exploration_noise = 30.  # @param {type:"number"}
final_exploration_noise = 20.  # @param {type:"number"}
exploration_noise = starting_exploration_noise  # not a param

actor_fc_layer_params = (64, 64)  # @param
actor_dropout_layer_params = None  # @param
critic_observation_fc_layer_params = None  # @param
critic_action_fc_layer_params = None  # @param
critic_joint_fc_layer_params = (64, 64)  # @param
critic_joint_dropout_layer_params = None  # @param

target_update_tau_p = 0.1  # @param {type:"number"}
target_update_period_p = 2  # @param {type:"integer"}
actor_update_period_p = 2  # @param {type:"integer"}
actor_learning_rate = 0.001  # @param{type:"number"}
critic_learning_rate = 0.001  # @param{type:"number"}

target_policy_noise_p = 30.  # @param {type:"number"}
target_policy_noise_clip_p = 75.  # @param {type:"number"}

learning_nodes = False  # @param {type:"boolean"}

auction_type = ['MultiLanes', 'SingleLane'][1]

weights_file_name = None if learning_nodes else 'weights_' + auction_type + '_' + str(node_auction_cost) + '_' + \
                                                str(n_carriers_per_node) + '.json'

e, learning_agent = load_tfa_env_and_agent(n_carriers=11 * n_carriers_per_node,  # 11 is the number of nodes
                                           shippers_reserve_price_per_distance=shippers_reserve_price_per_distance,
                                           init_node_weights_distance_scaling_factor=init_node_weights_distance_scaling_factor,
                                           max_node_weights_distance_scaling_factor=max_node_weights_distance_scaling_factor,
                                           shipper_default_reserve_price=shipper_default_reserve_price,
                                           node_auction_cost=node_auction_cost,
                                           auction_type=auction_type,
                                           node_nb_info=node_nb_info,
                                           learning_nodes=learning_nodes,
                                           weights_file_name=weights_file_name,
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
                                           critic_joint_dropout_layer_params=critic_joint_dropout_layer_params,
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
               'nb_discarded_loads': [],
               'nb_in_transit_loads': [],
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
            carriers_profit.append(sum(carrier_p.episode_revenues[1:]) - sum(carrier_p.episode_expenses[1:]))
        else:
            carriers_profit.append(0.)
    carriers_profit = np.array(carriers_profit)

    nb_loads = len(e.loads)
    nb_arrived_loads = 0
    nb_discarded_loads = 0
    nb_in_transit_loads = 0
    total_delivery_costs = []
    nb_hops = []
    delivery_times = []
    for load_p in e.loads:
        if load_p.is_arrived:
            nb_arrived_loads += 1
            total_delivery_costs.append(load_p.total_delivery_cost())
            nb_hops.append(load_p.nb_hops())
            delivery_times.append(load_p.delivery_time())
        if load_p.is_discarded:
            nb_discarded_loads += 1
        if load_p.in_transit:
            nb_in_transit_loads += 1
    total_delivery_costs = np.array(total_delivery_costs)
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
               'nb_discarded_loads': nb_discarded_loads,
               'nb_in_transit_loads': nb_in_transit_loads,
               'delivery_costs': {'min': np.min(total_delivery_costs),
                                  'quartile1': np.quantile(total_delivery_costs, 0.25),
                                  'quartile2': np.quantile(total_delivery_costs, 0.5),
                                  'quartile3': np.quantile(total_delivery_costs, 0.75),
                                  'max': np.max(total_delivery_costs),
                                  'mean': np.mean(total_delivery_costs)},
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
keys_without_stat = ['nb_loads', 'nb_arrived_loads', 'nb_discarded_loads', 'nb_in_transit_loads']
stat_keys = ['min', 'quartile1', 'quartile2', 'quartile3', 'max', 'mean']


def add_results(results) -> None:
    for key_with_stats in keys_with_stats:
        for stat_key in stat_keys:
            all_results[key_with_stats][stat_key].append(results[key_with_stats][stat_key])
    for key_without_stat in keys_without_stat:
        all_results[key_without_stat].append(results[key_without_stat])


"""## Loop"""

num_rounds = 25  # @param {type:"integer"}
num_cost_pass = 10  # @param {type:"integer"}
num_train_per_pass = 20  # @param {type:"integer"}
num_iteration_per_test = 50  # @param{type:"integer"}

exploration_noise_update = (starting_exploration_noise - final_exploration_noise) / (num_rounds - 1)


def change_costs():
    for carrier_p in learning_agent.carriers:
        carrier_p.random_new_cost_parameters()


start_time = time.time()
for i in range(num_rounds):
    now = time.time()
    if i > 0:
        eta = int((now - start_time)*(num_rounds-i)/i)
        eta_h = eta // 3600
        eta_m = (eta % 3600) // 60
        eta_s = (eta % 3600) % 60
        print("ETA:", "{}h{}m{}s".format(eta_h, eta_m, eta_s))
    print("Test", i+1, '/', num_rounds)
    change_costs()
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
            for _ in range(n):  # do not parallelize
                carrier = e.pop_enough_transitions_carriers()
                experience, _ = next(carrier.training_data_set_iter)
                train(experience=experience, weights=None)
    exploration_noise -= exploration_noise_update
    learning_agent.change_exploration_noise_std(exploration_noise)

print("Final test")
change_costs()
test_results = test(num_iteration_per_test)
print(test_results)
add_results(test_results)

end = time.time()
total_time = int(end - start_time)
total_time_h = total_time//3600
total_time_m = (total_time%3600)//60
total_time_s = (total_time%3600)%60
print("Total time:", "{}h{}m{}s".format(total_time_h, total_time_m, total_time_s))
