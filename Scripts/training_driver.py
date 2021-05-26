#!/usr/bin/env python

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
import numpy as np

n_carriers_per_node = 15  # @param {type:"integer"}
discount = 0.95  # @param {type:"number"}

starting_exploration_noise = 500  # @param {type:"number"}
final_exploration_noise = 30  # @param {type:"number"}
exploration_noise = starting_exploration_noise

target_update_tau_p = 0.1  # @param {type:"number"}
target_update_period_p = 2  # @param {type:"number"}
actor_update_period_p = 2  # @param {type:"integer"}

reward_scale_factor_p = 1 / 500  # do not change, keep fixed at 500

target_policy_noise_p = 30.  # @param {type:"number"}
target_policy_noise_clip_p = 75.  # @param {type:"number"}

max_time_not_at_home = 30  # @param {type:"integer"}

e, learning_agent = load_env_and_agent(n_carriers=11 * n_carriers_per_node,
                                       discount=discount,
                                       exploration_noise=starting_exploration_noise,
                                       target_update_tau_p=target_update_tau_p,
                                       target_update_period_p=target_update_period_p,
                                       actor_update_period_p=actor_update_period_p,
                                       reward_scale_factor_p=reward_scale_factor_p,
                                       target_policy_noise_p=target_policy_noise_p,
                                       target_policy_noise_clip_p=target_policy_noise_clip_p,
                                       max_time_not_at_home=max_time_not_at_home)

train = tfa_function(learning_agent.train)


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
        print("Test iteration", counter, "/", num_iter_per_test - 1)
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

keys_with_stats = ['carriers_profit', 'delivery_costs', 'nb_hops', 'delivery_times']
keys_without_stat = ['nb_loads', 'nb_arrived_loads']
stat_keys = ['min', 'quartile1', 'quartile2', 'quartile3', 'max', 'mean']


def add_results(results) -> None:
    for key_with_stats in keys_with_stats:
        for stat_key in stat_keys:
            all_results[key_with_stats][stat_key].append(results[key_with_stats][stat_key])
    for key_without_stat in keys_without_stat:
        all_results[key_without_stat].append(results[key_without_stat])


num_rounds = 25  # @param {type:"integer"}
num_cost_pass = 4  # @param {type:"integer"}
num_train_per_pass = 2  # @param {type:"integer"}
num_iteration_before_train = 3  # @param {type:"integer"}
num_iteration_per_test = 3  # @param{type:"integer"}

exploration_noise_update = (starting_exploration_noise - final_exploration_noise) / (num_rounds - 1)


def change_costs():
    for carrier_p in learning_agent.carriers:
        carrier_p.random_new_cost_parameters()


# initialize the test lists to []
# add an ETA
for i in range(num_rounds):
    print("Test", i, '/', num_rounds - 1)
    change_costs()
    test_results = test(num_iteration_per_test)
    print(test_results)
    add_results(test_results)
    for j in range(num_cost_pass):
        print("Pass", j, "/", num_cost_pass - 1)
        change_costs()
        for k in range(num_train_per_pass + num_iteration_before_train):
            print("Training",
                  k, "/",
                  num_train_per_pass + num_iteration_before_train - 1,
                  "(including {1} preparation iteration)".format(num_train_per_pass, num_iteration_before_train))
            e.iteration()
            e.shuffle_new_transition_carriers()
            n = len(e.new_transition_carriers)
            if k > num_iteration_before_train-1:
                for _ in range(n):
                    carrier = e.pop_new_transition_carriers()
                    experience, _ = next(carrier.training_data_set_iter)
                    train(experience=experience, weights=None)
    exploration_noise -= exploration_noise_update
    learning_agent.change_exploration_noise_std(exploration_noise)
