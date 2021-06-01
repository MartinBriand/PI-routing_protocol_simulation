"""Loaders common to all games"""

import os, csv
import numpy as np
from typing import List, Dict
import random

from PI_RPS.Mechanics.Actors.Nodes.dummy_node import DummyNode, DummyNodeWeightMaster
from PI_RPS.Mechanics.Actors.Nodes.node import Node
from PI_RPS.Mechanics.Actors.Shippers.dummy_shipper import DummyShipper
from PI_RPS.Mechanics.Actors.Shippers.shipper import Shipper, NodeLaw
from PI_RPS.Mechanics.Environment.environment import Environment
from PI_RPS.Mechanics.Environment.tfa_environment import TFAEnvironment
from PI_RPS.Mechanics.Tools.load import Load


nb_hours_per_time_unit: float = 6.147508  # 390 km at an average speed of 39.42 km/h)
t_c_mu: float = 33. * nb_hours_per_time_unit
t_c_sigma: float = 4.15 * nb_hours_per_time_unit
ffh_c_mu: float = 20. * nb_hours_per_time_unit
ffh_c_sigma: float = 1.00 * nb_hours_per_time_unit


def load_realistic_nodes_and_shippers_to_env(e: Environment,
                                             node_nb_info: int,
                                             shippers_reserve_price_per_distance: float,
                                             shipper_default_reserve_price: float,
                                             ) -> None:
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

    # create Nodes
    weight_master = DummyNodeWeightMaster(environment=e,
                                          nb_infos=node_nb_info)
    for name in lambdas.keys():
        DummyNode(name=name,
                  weight_master=weight_master,
                  revenues=[],
                  environment=e)

    if isinstance(e, TFAEnvironment):
        e.build_node_state()

    lambdas, attribution, distances = _to_node_keys(e, lambdas, attribution, distances)
    e.set_distances(distances)

    weight_master.initialize()

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
