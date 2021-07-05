"""
This is a learning script to learn the weights of the game with non learning carriers
"""

import numpy as np
import random
import time

from PI_RPS.Games.init_tools import load_realistic_nodes_and_shippers_to_env, save_cost_learning_game
from PI_RPS.Games.init_tools import nb_hours_per_time_unit, t_c_mu, t_c_sigma, ffh_c_mu, ffh_c_sigma
from PI_RPS.Mechanics.Actors.Carriers.learning_cost_carrier import MultiLanesLearningCostsCarrier, \
    SingleLaneLearningCostsCarrier
from PI_RPS.Mechanics.Environment.environment import Environment

node_filter = ['Bremen', 'Dresden', 'Madrid', 'Marseille', 'Milan', 'Naples', 'Paris', 'Rotterdam', 'Saarbrücken',
               'Salzburg', 'Warsaw']

n_carriers_per_node = 30  # @param {type:"integer"}
max_nb_infos_per_node = 20
nb_lives_start = 8
nb_lives_after = 15

shippers_reserve_price_per_distance = 1200.  # @param{type:"number"}
shipper_default_reserve_price = 10000.  # @param{type:"number"}
init_node_weights_distance_scaling_factor = 500.  # @param{type:"number"}
initial_cost_majoration = 1.5
# not used if initialized by artificial weights
max_node_weights_distance_scaling_factor = 500. * 1.3  # @param{type:"number"}
# should be big enough to be unrealistic.
node_auction_cost = 0.  # @param{type:"number"}
node_nb_info = 100  # @param{type:"integer"}
max_nb_infos_per_load = 15  # @param{type:"integer"}

max_lost_auctions_in_a_row = 5  # @param {type:"integer"}
max_time_not_at_home = 24  # about 6 days before getting back home

auction_type = ['MultiLanes', 'SingleLane'][0]

learning_nodes = False  # @param{type:"boolean"}

weights_file_name = None if learning_nodes or auction_type == 'SingleLane' else \
    'weights_MultiLanes_' + str(node_auction_cost) + '_' + \
    str(n_carriers_per_node) + '_' + str(initial_cost_majoration) + '.json'

terminaison_file_name = 1
saving_file_name = auction_type + '_' + str(node_auction_cost) + '_' + str(terminaison_file_name) + '.bin'

# Initialize the environment
e = Environment(nb_hours_per_time_unit=nb_hours_per_time_unit,
                max_nb_infos_per_load=max_nb_infos_per_load,
                init_node_weights_distance_scaling_factor=init_node_weights_distance_scaling_factor,
                max_node_weights_distance_scaling_factor=max_node_weights_distance_scaling_factor,
                t_c_mu=t_c_mu,
                t_c_sigma=t_c_sigma,
                ffh_c_mu=ffh_c_mu,
                ffh_c_sigma=ffh_c_sigma, )

# Initialize the nodes and the shippers with the initial weights for the node according to a file
load_realistic_nodes_and_shippers_to_env(e=e,
                                         node_filter=node_filter,
                                         node_nb_info=node_nb_info,
                                         shippers_reserve_price_per_distance=shippers_reserve_price_per_distance,
                                         shipper_default_reserve_price=shipper_default_reserve_price,
                                         node_auction_cost=node_auction_cost,
                                         learning_nodes=learning_nodes,
                                         weights_file_name=weights_file_name,
                                         auction_type=auction_type
                                         )

# Get a reference to the weight master of the nodes, this will help when training the nodes
weight_master = e.nodes[0].weight_master

# Create the carriers
counter = {}


def create_carrier(node_p, nb_lives_p):
    if node_p in counter.keys():
        counter[node_p] += 1
    else:
        counter[node_p] = 1
    road_costs = random.normalvariate(mu=t_c_mu, sigma=t_c_sigma)
    drivers_costs = random.normalvariate(mu=ffh_c_mu, sigma=ffh_c_sigma)

    if auction_type == 'MultiLanes':
        MultiLanesLearningCostsCarrier(name=node_p.name + '_' + str(counter[node_p]),
                                       home=node_p,
                                       in_transit=False,
                                       previous_node=node_p,
                                       next_node=node_p,
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
                                       nb_lost_auctions_in_a_row=0,
                                       max_lost_auctions_in_a_row=max_lost_auctions_in_a_row,
                                       last_won_node=None,
                                       nb_episode_at_last_won_node=0,
                                       nb_lives=nb_lives_p,
                                       max_nb_infos_per_node=max_nb_infos_per_node,
                                       costs_table=None,
                                       list_of_costs_table=None,
                                       is_learning=True
                                       )
    elif auction_type == 'SingleLane':
        SingleLaneLearningCostsCarrier(name=node_p.name + '_' + str(counter[node_p]),
                                       home=node_p,
                                       in_transit=False,
                                       previous_node=node_p,
                                       next_node=node_p,
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
                                       nb_lost_auctions_in_a_row=0,
                                       max_lost_auctions_in_a_row=max_lost_auctions_in_a_row,
                                       last_won_node=None,
                                       nb_episode_at_last_won_node=0,
                                       nb_lives=nb_lives_p,
                                       max_nb_infos_per_node=max_nb_infos_per_node,
                                       costs_table=None,
                                       list_of_costs_table=None,
                                       is_learning=True
                                       )


for k in range(n_carriers_per_node * len(e.nodes)):
    node = e.nodes[k % len(e.nodes)]
    create_carrier(node, nb_lives_p=nb_lives_start)

# Result structure
all_results = {'type': [],
               'carriers_profit': {'min': [],
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


# Training loop
def clear_env() -> None:
    e.clear_node_auctions()
    e.clear_loads()
    e.clear_carrier_profits()
    e.clear_shipper_expenses()


def test(num_iter_per_test: int, loop_p: int, phase_p: int, sub_phase_p: int,
         test_nb_p: int, prop_reserve_price_involved_threshold_p: float):
    # clear
    clear_env()

    # Running environment
    for _ in range(num_iter_per_test):
        e.iteration()

    # Getting data
    type_p = (loop_p, phase_p, sub_phase_p, test_nb_p)
    carriers_profit = []
    carriers_with_non_positive_profit = []
    for carrier_p in e.carriers:
        if len(carrier_p.episode_revenues) > 1:
            profit = sum(carrier_p.episode_revenues[1:]) - sum(carrier_p.episode_expenses[1:])
        else:
            profit = 0.
        carriers_profit.append(profit)
        if profit <= 0.:
            carriers_with_non_positive_profit.append(carrier_p)
    carriers_profit = np.array(carriers_profit)

    nb_loads = len(e.loads)
    nb_arrived_loads = 0
    nb_discarded_loads = 0
    nb_in_transit_loads = 0
    total_delivery_costs = []
    nb_hops = []
    delivery_times = []
    nodes_nb_transaction = {}
    nodes_nb_transaction_with_reserve_price_involved = {}
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

        for movement in load_p.movements:
            if movement[0] not in nodes_nb_transaction.keys():  # movement[0] is the node where the auction occurred
                nodes_nb_transaction[movement[0]] = 0
                nodes_nb_transaction_with_reserve_price_involved[movement[0]] = 0

            nodes_nb_transaction[movement[0]] += 1
            if movement[5]:  # movement[5] is the reserve_price_involved of the auction
                nodes_nb_transaction_with_reserve_price_involved[movement[0]] += 1

    nodes_with_too_much_reserve_price = [node_p for node_p in nodes_nb_transaction.keys()
                                         if nodes_nb_transaction_with_reserve_price_involved[node_p]
                                         / nodes_nb_transaction[node_p] > prop_reserve_price_involved_threshold_p]

    total_delivery_costs = np.array(total_delivery_costs)
    nb_hops = np.array(nb_hops)
    delivery_times = np.array(delivery_times)

    results = {'type': type_p,
               'carriers_profit': {'min': np.min(carriers_profit),
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
    return results, carriers_with_non_positive_profit, nodes_with_too_much_reserve_price


keys_with_stats = ['carriers_profit', 'delivery_costs', 'nb_hops', 'delivery_times']
keys_without_stat = ['type', 'nb_loads', 'nb_arrived_loads', 'nb_discarded_loads', 'nb_in_transit_loads']
stat_keys = ['min', 'quartile1', 'quartile2', 'quartile3', 'max', 'mean']


def add_results(results) -> None:
    for key_with_stats in keys_with_stats:
        for stat_key in stat_keys:
            all_results[key_with_stats][stat_key].append(results[key_with_stats][stat_key])
    for key_without_stat in keys_without_stat:
        all_results[key_without_stat].append(results[key_without_stat])


# Loop parts
loop = 0
phase = 0
sub_phase = 0
test_nb = 0
num_iteration_per_test = 500
num_train_per_pass = 2000
carrier_convergence_nb_iter = 25
prop_reserve_price_involved_threshold = 0.01


# General loop iteration
def loop_fn():
    global num_iteration_per_test, num_train_per_pass
    global loop, phase, test_nb, prop_reserve_price_involved_threshold
    test_results, carriers_with_non_positive_profit, nodes_with_too_much_reserve_price = \
        test(num_iteration_per_test, loop, phase, sub_phase, test_nb, prop_reserve_price_involved_threshold)
    print(test_results)
    add_results(test_results)
    for _ in range(num_train_per_pass):
        e.iteration()

    test_nb += 1
    return carriers_with_non_positive_profit, nodes_with_too_much_reserve_price


# Getting carriers and nodes to converge (part1)
# getting the carriers to converge
def convergence_carrier():
    for _ in range(carrier_convergence_nb_iter):
        list_of_convergence_degree = [carrier_p.convergence_state() for carrier_p in e.carriers]
        print('Node level of knowledge:', sum(list_of_convergence_degree) / len(list_of_convergence_degree))
        loop_fn()


# getting the nodes to converge
def convergence_nodes():
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

    while len(not_converged.keys()) > 0:
        loop_fn()
        add_weights_to_lists()
        print('Not converged nodes:', not_converged)


def part1():
    global phase, sub_phase, test_nb
    phase = 0
    sub_phase = 0
    test_nb = 0
    # do not touch the node weights
    if weight_master.is_learning:  # set nodes to not learning
        weight_master.is_learning = False
    for carrier in e.carriers:  # set carriers to learning and reinit_absolutely
        carrier.reinit_cost_tables_to_0()
        if not carrier.is_learning:
            carrier.is_learning = True
    convergence_carrier()

    if auction_type == 'MultiLanes':
        sub_phase = 1
        test_nb = 0
        weight_master.reinitialize()
        if not weight_master.is_learning:  # set nodes to learning
            weight_master.is_learning = True
        for carrier in e.carriers:  # set carriers to not learning
            if carrier.is_learning:
                carrier.is_learning = False
        convergence_nodes()

        sub_phase = 2
        test_nb = 0
        # do not touch the node weights
        for carrier in e.carriers:  # set carriers to learning and reinit to average
            carrier.reinit_cost_tables_to_average()
            if not carrier.is_learning:
                carrier.is_learning = True
        convergence_carrier()


# Running the game until we have a stabilized market (part2)
def part2():
    global phase, sub_phase, test_nb
    phase = 1
    sub_phase = 0
    test_nb = 0
    nb_iter = 0
    # everyone is already learning because of phase 1
    carriers_with_non_positive_profit, nodes_with_too_much_reserve_price = loop_fn()
    old_nb_carriers = len(e.carriers)

    def part2_iter(carriers_with_non_positive_profit_p, nodes_with_too_much_reserve_price_p):
        for carrier in carriers_with_non_positive_profit_p:
            carrier.remove_a_life()
        for node_p in nodes_with_too_much_reserve_price_p:
            create_carrier(node_p=node_p, nb_lives_p=nb_lives_after)
        print(len(e.carriers), 'carriers', )
        print(len(carriers_with_non_positive_profit_p), 'carriers with non positive profit')
        print(len(nodes_with_too_much_reserve_price_p), 'nodes with too much reserve price')
        return loop_fn()

    while len(e.carriers) <= old_nb_carriers:
        old_nb_carriers = len(e.carriers)
        carriers_with_non_positive_profit, nodes_with_too_much_reserve_price = \
            part2_iter(carriers_with_non_positive_profit, nodes_with_too_much_reserve_price)
        nb_iter += 1
    if nb_iter > 1:
        print("15 more")
        for _ in range(15):
            carriers_with_non_positive_profit, nodes_with_too_much_reserve_price = \
                part2_iter(carriers_with_non_positive_profit, nodes_with_too_much_reserve_price)
    return nb_iter


# Loop
start_time = time.time()
part1()
part2()
loop += 1
part1()
end_time = time.time()
delta = int(end_time - start_time)
delta_h = delta // 3600
delta_m = (delta % 3600) // 60
delta_s = (delta % 3600) % 60
save_cost_learning_game(e, saving_file_name)
print("Total time:", "{}h{}m{}s".format(delta_h, delta_m, delta_s))
