"""

"""
# TODO write description


import numpy as np
import random
import time

from PI_RPS.Games.init_tools import load_realistic_nodes_and_shippers_to_env
from PI_RPS.Games.init_tools import nb_hours_per_time_unit, t_c_mu, t_c_sigma, ffh_c_mu, ffh_c_sigma
from PI_RPS.Mechanics.Actors.Carriers.cost_bidding_carrier import CostBiddingCarrier
from PI_RPS.Mechanics.Environment.environment import Environment


n_carriers_per_node = 15  # @param {type:"integer"}

shippers_reserve_price_per_distance = 1200.  # @param{type:"number"}
shipper_default_reserve_price = 20000.  # @param{type:"number"}
init_node_weights_distance_scaling_factor = 500  # @param{type:"number"}
max_node_weights_distance_scaling_factor = 500  # @param{type:"number"}
# should be big enough to be unrealistic.
node_nb_info = 100  # @param{type:"integer"}
max_nb_infos_per_load = 15  # @param{type:"integer"}

max_time_not_at_home = 30  # @param {type:"integer"}

e = Environment(nb_hours_per_time_unit=nb_hours_per_time_unit,
                max_nb_infos_per_load=max_nb_infos_per_load,
                init_node_weights_distance_scaling_factor=init_node_weights_distance_scaling_factor,
                max_node_weights_distance_scaling_factor=max_node_weights_distance_scaling_factor)

load_realistic_nodes_and_shippers_to_env(e=e,
                                         node_nb_info=node_nb_info,
                                         shippers_reserve_price_per_distance=shippers_reserve_price_per_distance,
                                         shipper_default_reserve_price=shipper_default_reserve_price)

counter = {}
for k in range(n_carriers_per_node * len(e.nodes)):
    node = e.nodes[k % len(e.nodes)]
    if node in counter.keys():
        counter[node] += 1
    else:
        counter[node] = 1
    road_costs = random.normalvariate(mu=t_c_mu, sigma=t_c_sigma)
    drivers_costs = random.normalvariate(mu=ffh_c_mu, sigma=ffh_c_sigma)

    CostBiddingCarrier(name=node.name + '_' + str(counter[node]),
                       home=node,
                       in_transit=False,
                       next_node=node,
                       time_to_go=0,
                       load=None,
                       environment=e,
                       episode_expenses=[],
                       episode_revenues=[],
                       this_episode_expenses=[],
                       this_episode_revenues=0,
                       transit_cost=road_costs,
                       far_from_home_cost=drivers_costs,
                       time_not_at_home=0,
                       max_time_not_at_home=max_time_not_at_home,
                       t_c_mu=t_c_mu,
                       t_c_sigma=t_c_sigma,
                       ffh_c_mu=ffh_c_mu,
                       ffh_c_sigma=ffh_c_sigma,
                       too_high_bid=shipper_default_reserve_price)

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


def clear_env() -> None:
    e.clear_node_auctions()
    e.clear_loads()
    e.clear_carrier_profits()
    e.clear_shipper_expenses()


def test(num_iter_per_test):
    # clear
    clear_env()

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
    clear_env()
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

num_rounds = 300  # @param {type:"integer"}
num_cost_pass = 5  # @param {type:"integer"}
num_train_per_pass = 100  # @param {type:"integer"}
num_iteration_per_test = 100  # @param{type:"integer"}


def change_costs():
    for carrier_p in e.carriers:
        carrier_p.random_new_cost_parameters()


start_time = time.time()
for i in range(num_rounds):
    now = time.time()
    if i > 0:
        eta = int((now - start_time) * (num_rounds - i) / i)
        eta_h = eta // 3600
        eta_m = (eta % 3600) // 60
        eta_s = (eta % 3600) % 60
        print("ETA:", "{}h{}m{}s".format(eta_h, eta_m, eta_s))
    print("Test", i + 1, '/', num_rounds)
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

print("Final test")
change_costs()
print(e.nodes[0].readable_weights())
test_results = test(num_iteration_per_test)
print(test_results)
add_results(test_results)
