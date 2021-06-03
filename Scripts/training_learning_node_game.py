"""
This is a learning script to learn the weights of the game with non learning carriers
"""


import numpy as np
import random
import time

from PI_RPS.Games.init_tools import load_realistic_nodes_and_shippers_to_env, write_readable_weights_json
from PI_RPS.Games.init_tools import nb_hours_per_time_unit, t_c_mu, t_c_sigma, ffh_c_mu, ffh_c_sigma
from PI_RPS.Mechanics.Actors.Carriers.cost_bidding_carrier import CostBiddingCarrier
from PI_RPS.Mechanics.Environment.environment import Environment

n_carriers_per_node = 25  # @param {type:"integer"}

shippers_reserve_price_per_distance = 1200.  # @param{type:"number"}
shipper_default_reserve_price = 20000.  # @param{type:"number"}
init_node_weights_distance_scaling_factor = 500.  # @param{type:"number"}
# not used if initialized by artificial weights
max_node_weights_distance_scaling_factor = 500. * 1.3  # @param{type:"number"}
# should be big enough to be unrealistic.
node_auction_cost = 0.  # @param{type:"number"}
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
                                         shipper_default_reserve_price=shipper_default_reserve_price,
                                         node_auction_cost=node_auction_cost,
                                         weights_file_name=None)

weight_master = e.nodes[0].weight_master

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
                       previous_node=node,
                       next_node=node,
                       time_to_go=0,
                       load=None,
                       environment=e,
                       episode_types=[],
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
    for _ in range(num_iter_per_test):
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

num_cost_pass = 5  # @param {type:"integer"}
num_train_per_pass = 100  # @param {type:"integer"}
num_iteration_per_test = 100  # @param{type:"integer"}


def change_costs():
    for carrier_p in e.carriers:
        carrier_p.random_new_cost_parameters()


init_readable_weights = weight_master.readable_weights()

not_converged = {arrival: [departure
                           for departure in init_readable_weights[arrival].keys() if departure != arrival]
                 for arrival in init_readable_weights.keys()}

previous_weights = {arrival: {departure: [init_readable_weights[arrival][departure]]
                              for departure in init_readable_weights[arrival].keys()}
                    for arrival in init_readable_weights.keys()}


def add_weights_to_lists():
    readable_weights = weight_master.readable_weights()
    for arrival in readable_weights.keys():
        for departure in readable_weights[arrival].keys():
            previous_weights[arrival][departure].append(readable_weights[arrival][departure])
    
    has_converged = {}
    for arrival in not_converged.keys():
        has_converged[arrival] = []
        for departure in not_converged[arrival]:
            this_previous_weights = previous_weights[arrival][departure]
            if this_previous_weights[-2] < this_previous_weights[-1]:
                has_converged[arrival].append(departure)
    for arrival in has_converged.keys():
        for departure in has_converged[arrival]:
            not_converged[arrival].remove(departure)
        if len(not_converged[arrival]) == 0:
            del not_converged[arrival]
            

start_time = time.time()
loop_counter = 0


def loop_fn(loop_counter):
    print("Test", loop_counter + 1)
    change_costs()
    print(weight_master.readable_weights())
    print(not_converged)
    test_results = test(num_iteration_per_test)
    print(test_results)
    add_results(test_results)
    for _ in range(num_cost_pass):
        change_costs()
        for _ in range(num_train_per_pass):
            e.iteration()
    add_weights_to_lists()


while len(not_converged.keys()) > 0:
    loop_fn(loop_counter)
    loop_counter += 1
print("Converged !!")
print("15 more for better convergence")
for _ in range(15):
    loop_fn(loop_counter)
    loop_counter += 1

end_time = time.time()
delta = int(end_time - start_time)
delta_h = delta // 3600
delta_m = (delta % 3600) // 60
delta_s = (delta % 3600) % 60
final_readable_weights = weight_master.readable_weights()
write_readable_weights_json(final_readable_weights, 'weights_' + str(node_auction_cost) + '_' +
                            str(n_carriers_per_node) + '.json')
print("Total time:", "{}h{}m{}s".format(delta_h, delta_m, delta_s))
