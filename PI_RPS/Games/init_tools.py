"""Loaders common to all games"""

import csv
import json
import os
import pickle
import random

import numpy as np
from typing import List, Dict, Optional, Type, Any

from PI_RPS.Mechanics.Actors.Carriers.learning_cost_carrier import SingleLaneLearningCostsCarrier, \
    MultiLanesLearningCostsCarrier, LearningCostsCarrier
from PI_RPS.Mechanics.Actors.Nodes.dummy_node import DummyNode, DummyNodeWeightMaster
from PI_RPS.Mechanics.Actors.Nodes.node import Node
from PI_RPS.Mechanics.Actors.Shippers.dummy_shipper import DummyShipper
from PI_RPS.Mechanics.Actors.Shippers.shipper import Shipper, NodeLaw
from PI_RPS.Mechanics.Environment.environment import Environment
from PI_RPS.Mechanics.Environment.tfa_environment import TFAEnvironment
from PI_RPS.Mechanics.Tools.load import Load
from PI_RPS.prj_typing.types import NodeWeights

nb_hours_per_time_unit: float = 6.147508  # 390 km at an average speed of 39.42 km/h)
t_c_mu: float = 33. * nb_hours_per_time_unit
t_c_sigma: float = 4.15 * nb_hours_per_time_unit
ffh_c_mu: float = 20. * nb_hours_per_time_unit
ffh_c_sigma: float = 1.00 * nb_hours_per_time_unit


def load_realistic_nodes_and_shippers_to_env(e: Environment,
                                             node_nb_info: int,
                                             shippers_reserve_price_per_distance: float,
                                             shipper_default_reserve_price: float,
                                             node_filter: List[str],
                                             node_auction_cost: float,
                                             auction_type: str,
                                             learning_nodes: bool,
                                             weights_file_name: str = None,
                                             weights_dict: Dict[str, float] = None
                                             ) -> None:
    path = os.path.abspath(os.path.dirname(__file__))
    lambdas: np.ndarray = _read_csv(os.path.join(path, 'data/city_traffic_lambda_table.csv'))
    attribution: np.ndarray = _read_csv(os.path.join(path, 'data/city_traffic_dest_attribution_table.csv'))
    distances: np.ndarray = _read_csv(os.path.join(path, 'data/city_distance_matrix_time_step.csv'))
    assert weights_dict is None or weights_file_name is None, "Can't set weights with two attributes"
    weights = _read_weights_json(weights_file_name) if weights_file_name else None
    weights = weights_dict if weights_dict else weights

    # here we filter everything except weights, which should be already filtered (or None)
    if node_filter is not None:
        lambdas, attribution, distances = _filter(node_filter, lambdas, attribution, distances)

    # check size
    lts = lambdas.shape
    ats = attribution.shape
    dts = distances.shape
    fs = len(node_filter)
    assert fs == lts[0], "Some of the names in the filter are wrong..."
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

    # check keys and lengths at the same time for weights
    if weights:
        level1_keys = np.array(list(weights.keys()))
        assert (lambdas[:, 0] == level1_keys).sum() == n, \
            "keys do not match:\n{}\n{}".format(lambdas[:, 0], level1_keys)
        for key1 in level1_keys:
            level2_keys = np.array(list(weights[key1].keys()))
            assert (level1_keys == level2_keys).sum() == n, \
                "keys do not match:\n{}\n{}".format(level1_keys, level2_keys)

    # make dicts
    lambdas, attribution, distances = _to_dicts(lambdas[:, 0], lambdas, attribution, distances)

    # create Nodes
    weight_master = DummyNodeWeightMaster(environment=e,
                                          nb_infos=node_nb_info,
                                          is_learning=learning_nodes
                                          )
    for name in lambdas.keys():
        DummyNode(name=name,
                  weight_master=weight_master,
                  revenues=[],
                  environment=e,
                  auction_cost=node_auction_cost,
                  auction_type=auction_type)

    if isinstance(e, TFAEnvironment):
        e.build_node_state()

    lambdas, attribution, distances, weights = _to_node_keys(e, lambdas, attribution, distances, weights)
    e.set_distances(distances)

    weight_master.initialize(weights)

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
            environment: Environment,
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


def _read_csv(file_path: str) -> np.ndarray:
    """Return a List with all the values"""
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        final: List = []
        for line in reader:
            final.append(line)
    return np.array(final)


def _filter(node_filter, lambdas, attribution, distances):
    new_lambdas = np.array([item for item in lambdas if item[0] in node_filter])

    def att_or_dist_filter(table_p):
        res = []
        for i in range(len(table_p)):
            row = []
            if i == 0 or table_p[i, 0] in node_filter:
                for j in range(len(table_p[i])):
                    if i == 0 and j == 0:
                        row.append(table_p[i, j])
                    elif j == 0 or table_p[0, j] in node_filter:
                        row.append(table_p[i, j])
                res.append(row)
        return np.array(res)

    return new_lambdas, att_or_dist_filter(attribution), att_or_dist_filter(distances)


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


def _to_node_keys(e: Environment,
                  lambdas: Dict[str, float],
                  attribution: Dict[str, Dict[str, float]],
                  distances: Dict[str, Dict[str, int]],
                  weights) -> (Dict[Node, float],
                               Dict[Node, Dict[Node, float]],
                               Dict[Node, Dict[Node, int]],
                               Optional[NodeWeights]):
    node_name_dict = {node.name: node for node in e.nodes}
    new_lambdas = {node_name_dict[name]: lamb for name, lamb in lambdas.items()}
    new_attribution = {node_name_dict[name1]: {node_name_dict[name2]: att for name2, att in obj.items()}
                       for name1, obj in attribution.items()}
    new_distances = {node_name_dict[name1]: {node_name_dict[name2]: dist for name2, dist in obj.items()}
                     for name1, obj in distances.items()}
    if weights:
        new_weights = {node_name_dict[name1]: {node_name_dict[name2]: att for name2, att in obj.items()}
                       for name1, obj in weights.items()}
    else:
        new_weights = None

    return new_lambdas, new_attribution, new_distances, new_weights


def _read_weights_json(file_name) -> Dict:
    """Read from a json"""
    path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(path, 'data/experimental/' + file_name)
    with open(path, 'r') as f:
        return json.load(f)


def write_readable_weights_json(readable_weights, file_name) -> None:
    """"Write weights to json"""
    path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(path, 'data/experimental/' + file_name)
    with open(path, 'w') as f:
        json.dump(readable_weights, f)


def load_learned_games(file_name: str) -> Environment:
    path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(path, 'game_configs/' + file_name)
    with open(path, 'rb') as f:
        d = pickle.load(f)
    e = Environment(nb_hours_per_time_unit=nb_hours_per_time_unit,
                    max_nb_infos_per_load=d['max_nb_infos_per_load'],
                    init_node_weights_distance_scaling_factor=d['init_node_weights_distance_scaling_factor'],
                    max_node_weights_distance_scaling_factor=d['max_node_weights_distance_scaling_factor'],
                    t_c_mu=t_c_mu,
                    t_c_sigma=t_c_sigma,
                    ffh_c_mu=ffh_c_mu,
                    ffh_c_sigma=ffh_c_sigma, )

    load_realistic_nodes_and_shippers_to_env(e=e,
                                             node_filter=d['node_filter'],
                                             node_nb_info=d['node_nb_info'],
                                             shippers_reserve_price_per_distance=d[
                                                 'shippers_reserve_price_per_distance'],
                                             shipper_default_reserve_price=d['shipper_default_reserve_price'],
                                             node_auction_cost=d['node_auction_cost'],
                                             learning_nodes=False,
                                             weights_dict=d['weights_dict'],
                                             auction_type=d['auction_type']
                                             )

    node_name_dict = {node.name: node for node in e.nodes}

    def transform_carrier_node_kwargs(carrier_args: Dict[str, Any]) -> Dict[str, Any]:
        to_transform_keys = ['home', 'previous_node', 'next_node']
        to_transform_dicts = ['costs_table', 'list_of_costs_table']

        return1 = {key: node_name_dict[value] for key, value in carrier_args.items()
                   if key in to_transform_keys}
        return2 = {key1: {node_name_dict[key2]: value2
                          for key2, value2 in value1.items()}
                   for key1, value1 in carrier_args.items() if key1 in to_transform_dicts}
        return3 = {key: value for key, value in carrier_args.items()
                   if key not in to_transform_keys+to_transform_dicts}

        return {**return1, **return2, **return3, **{'environment': e}}

    for carrier_config in d['carriers']:
        if d['auction_type'] == 'SingleLane':
            if carrier_config['type'] == 'CostLearning':
                SingleLaneLearningCostsCarrier(**transform_carrier_node_kwargs(carrier_config['kwargs']))
            else:
                raise NotImplementedError
        elif d['auction_type'] == 'MultiLanes':
            if carrier_config['type'] == 'CostLearning':
                MultiLanesLearningCostsCarrier(**transform_carrier_node_kwargs(carrier_config['kwargs']))
            else:
                raise NotImplementedError

    return e


def save_cost_learning_game(e: Environment, file_name: str) -> None:
    d = {'max_nb_infos_per_load': e.max_nb_infos_per_load,
         'init_node_weights_distance_scaling_factor': e.init_node_weights_distance_scaling_factor,
         'max_node_weights_distance_scaling_factor': e.max_node_weights_distance_scaling_factor,
         'node_filter': [node.name for node in e.nodes], 'node_nb_info': e.nodes[0].weight_master.nb_infos,
         'shippers_reserve_price_per_distance': e.shippers[0].reserve_price_per_distance,
         'shipper_default_reserve_price': e.shippers[0].default_reserve_price,
         'node_auction_cost': e.nodes[0].auction_cost(), 'weights_dict': e.nodes[0].weight_master.weights_text(),
         'auction_type': e.nodes[0].auction_type}

    carrier_configs = []
    for carrier in e.carriers:
        if isinstance(carrier, LearningCostsCarrier):
            config = {'type': 'CostLearning',
                      'kwargs': {'name': carrier.name,
                                 'home': carrier.home.name,
                                 'in_transit': False,
                                 'previous_node': carrier.home.name,
                                 'next_node': carrier.home.name,
                                 'time_to_go': 0,
                                 'load': None,
                                 'episode_types': [],
                                 'episode_expenses': [],
                                 'episode_revenues': [],
                                 'this_episode_expenses': [],
                                 'this_episode_revenues': 0,
                                 'transit_cost': carrier.t_c,
                                 'far_from_home_cost': carrier.ffh_c,
                                 'time_not_at_home': 0,
                                 'nb_lost_auctions_in_a_row': 0,
                                 'max_lost_auctions_in_a_row': carrier.max_lost_auctions_in_a_row,
                                 'last_won_node': None,
                                 'nb_episode_at_last_won_node': 0,
                                 'nb_lives': carrier.nb_lives,
                                 'max_nb_infos_per_node': carrier.max_nb_infos_per_node,
                                 'costs_table': {key.name: value
                                                 for key, value in carrier.cost_table.items()},
                                 'list_of_costs_table': {key.name: value
                                                         for key, value in carrier.list_of_costs_table.items()},
                                 'is_learning': False
                                 }
                      }
        else:
            raise NotImplementedError
        carrier_configs.append(config)

    d['carriers'] = carrier_configs

    path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(path, 'game_configs/' + file_name)
    with open(path, 'wb') as f:
        pickle.dump(d, f)
