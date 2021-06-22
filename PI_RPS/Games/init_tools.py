"""Loaders common to all games"""

import os, csv, json
import numpy as np
from typing import List, Dict, Optional
import random

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
                                             weights_file_name: str = None
                                             ) -> None:
    path = os.path.abspath(os.path.dirname(__file__))
    lambdas: np.ndarray = _read_csv(os.path.join(path, 'data/city_traffic_lambda_table.csv'))
    attribution: np.ndarray = _read_csv(os.path.join(path, 'data/city_traffic_dest_attribution_table.csv'))
    distances: np.ndarray = _read_csv(os.path.join(path, 'data/city_distance_matrix_time_step.csv'))
    weights = _read_weights_json(weights_file_name) if weights_file_name else None

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
