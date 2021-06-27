# -*- coding: utf-8 -*-
"""
Training script for the learning carriers (still with bugs for the moment)
Now we learn from whole episode
"""


from tf_agents.utils.common import function as tfa_function
from PI_RPS.Games.Learning_Game.initialize_episode import load_tfa_env_and_agent
import numpy as np
import time

"""# Initialization"""
node_filter = ['Bremen', 'Dresden']  # , 'Madrid', 'Marseille', 'Milan', 'Naples', 'Paris', 'Rotterdam', 'Saarbrücken',
              # 'Salzburg', 'Warsaw']

n_carriers_per_node = 15  # @param {type:"integer"}
cost_majoration_file = 1.  # to select the correct weights  @param {type:"integer"}
action_min = 0.  # @param {type:"number"}
action_max = 10.  # @param {type:"number"}

shippers_reserve_price_per_distance = 1200.  # @param{type:"number"}
shipper_default_reserve_price = 10000.  # @param{type:"number"}

init_node_weights_distance_scaling_factor = None  # 500.  # @param{type:"number"}
max_node_weights_distance_scaling_factor = None  # 500. * 1.3  # @param{type:"number"}
# should be big enough to be unrealistic.
# They won't be used if not learning nodes

node_auction_cost = 0.  # @param{type:"number"}
node_nb_info = 100  # @param{type:"integer"}
# not used if node is not learning

max_nb_infos_per_load = 5  # @param{type:"integer"}
# not used if not learning nodes

max_lost_auctions_in_a_row = 5  # @param {type:"integer"}

reward_scale_factor_p = 1. / 20000.  # keep at 1./500., not a parameter

replay_buffer_batch_size = 8  # @param {type:"integer"}
buffer_max_length = 10  # @param{type:"integer"}

starting_exploration_noise = 0.2  # @param {type:"number"}
final_exploration_noise = 0.02  # @param {type:"number"}
exploration_noise = starting_exploration_noise  # not a param

actor_fc_layer_params = (32, 32)  # @param
actor_dropout_layer_params = None  # @param
critic_observation_fc_layer_params = None  # @param
critic_action_fc_layer_params = None  # @param
critic_joint_fc_layer_params = (32, 32)  # @param
critic_joint_dropout_layer_params = None  # @param

target_update_tau_p = 0.1  # @param {type:"number"}
target_update_period_p = 2  # @param {type:"integer"}
actor_update_period_p = 2  # @param {type:"integer"}
actor_learning_rate = 0.005  # @param{type:"number"}
critic_learning_rate = 0.005  # @param{type:"number"}

target_policy_noise_p = final_exploration_noise  # @param {type:"number"}
target_policy_noise_clip_p = target_policy_noise_p * 75. * 30.  # not a parameter

learning_nodes = False  # @param {type:"boolean"}

auction_type = ['MultiLanes', 'SingleLane'][0]

weights_file_name = None if learning_nodes else 'weights_' + auction_type + '_' + str(node_auction_cost) + '_' + \
                                                str(n_carriers_per_node) + '_' + str(cost_majoration_file) + '.json'

weights_file_name = None if learning_nodes else 'B-D_' + auction_type + '_' + str(node_auction_cost) + '_' + \
                                                str(n_carriers_per_node) + '_' + str(cost_majoration_file) + '.json'

e, learning_agent = load_tfa_env_and_agent(n_carriers=len(node_filter) * n_carriers_per_node,
                                           shippers_reserve_price_per_distance=shippers_reserve_price_per_distance,
                                           init_node_weights_distance_scaling_factor=init_node_weights_distance_scaling_factor,
                                           max_node_weights_distance_scaling_factor=max_node_weights_distance_scaling_factor,
                                           shipper_default_reserve_price=shipper_default_reserve_price,
                                           node_filter=node_filter,
                                           node_auction_cost=node_auction_cost,
                                           auction_type=auction_type,
                                           node_nb_info=node_nb_info,
                                           learning_nodes=learning_nodes,
                                           weights_file_name=weights_file_name,
                                           max_nb_infos_per_load=max_nb_infos_per_load,
                                           exploration_noise=exploration_noise,
                                           target_update_tau_p=target_update_tau_p,
                                           target_update_period_p=target_update_period_p,
                                           actor_update_period_p=actor_update_period_p,
                                           reward_scale_factor_p=reward_scale_factor_p,
                                           target_policy_noise_p=target_policy_noise_p,
                                           target_policy_noise_clip_p=target_policy_noise_clip_p,
                                           max_lost_auctions_in_a_row=max_lost_auctions_in_a_row,
                                           action_min=action_min,
                                           action_max=action_max,
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
               'values': {'min': [],
                          'quartile1': [],
                          'quartile2': [],
                          'quartile3': [],
                          'max': [],
                          'mean': []},
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


def test(num_iter_per_test):
    # clear
    clear_env(start=True)

    # Running environment
    for counter in range(num_iter_per_test):
        e.iteration()

    # Getting data
    carriers_profit = []
    carriers_value = []
    for carrier_p in e.carriers:
        if len(carrier_p.episode_revenues) > 1:
            carriers_profit.append(sum(carrier_p.episode_revenues[1:]) - sum(carrier_p.episode_expenses[1:]))
            carriers_value.append(carrier_p.cost_majoration)
        else:
            carriers_profit.append(0.)
    carriers_profit = np.array(carriers_profit)
    carriers_value = np.array(carriers_value)

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
               'values': {'min': np.min(carriers_value),
                          'quartile1': np.quantile(carriers_value, 0.25),
                          'quartile2': np.quantile(carriers_value, 0.5),
                          'quartile3': np.quantile(carriers_value, 0.75),
                          'max': np.max(carriers_value),
                          'mean': np.mean(carriers_value)},
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


keys_with_stats = ['carriers_profit', 'values', 'delivery_costs', 'nb_hops', 'delivery_times']
keys_without_stat = ['nb_loads', 'nb_arrived_loads', 'nb_discarded_loads', 'nb_in_transit_loads']
stat_keys = ['min', 'quartile1', 'quartile2', 'quartile3', 'max', 'mean']


def add_results(results) -> None:
    for key_with_stats in keys_with_stats:
        for stat_key in stat_keys:
            all_results[key_with_stats][stat_key].append(results[key_with_stats][stat_key])
    for key_without_stat in keys_without_stat:
        all_results[key_without_stat].append(results[key_without_stat])


"""## Loop"""

num_rounds = 100  # @param {type:"integer"}
num_cost_pass = 10  # @param {type:"integer"}
num_train_per_pass = 3  # @param {type:"integer"}
num_iteration_per_test = 50  # @param{type:"integer"}
num_iteration_per_episode = num_iteration_per_test

exploration_noise_update = (starting_exploration_noise - final_exploration_noise) / (num_rounds - 1)


def change_costs():
    for carrier_p in learning_agent.carriers:
        carrier_p.random_new_cost_parameters()


def print_nice_time(t, prefix):
    t_h = t // 3600
    t_m = (t % 3600) // 60
    t_s = (t % 3600) % 60
    print("{}: {}h{}m{}s".format(prefix, t_h, t_m, t_s))


start_time = time.time()
for i in range(num_rounds):
    now = time.time()
    if i > 0:
        eta = int((now - start_time) * (num_rounds - i) / i)
        print_nice_time(eta, 'ETA')
    print("Test", i + 1, '/', num_rounds)
    change_costs()
    test_results = test(num_iteration_per_test)
    print(test_results)
    add_results(test_results)
    for j in range(num_cost_pass):
        # print("Pass", j+1, "/", num_cost_pass)
        change_costs()
        while not all([carrier.replay_buffer_is_full for carrier in learning_agent.carriers]):
            for _ in range(num_iteration_per_episode):
                e.iteration()
            for carrier in learning_agent.carriers:
                carrier.finish_this_step_and_prepare_next_step()

        for _ in range(num_train_per_pass):
            # print("Training", k+1, "/", num_train_per_pass)
            for carrier in learning_agent.carriers:
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
print_nice_time(total_time, "Total time")
