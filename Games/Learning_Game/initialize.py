"""
This files defines a few funciton to unitialize the variables in exploitation_driver.py and training_driver.py
"""

import csv
import numpy as np
from typing import TYPE_CHECKING, List, Dict

from Mechanics.Actors.nodes.dummy_node import DummyNode
from Mechanics.Actors.nodes.node import Node
from Mechanics.Environment.tfa_environment import TFAEnvironment


def load_env() -> 'TFAEnvironment':
    path = ""
    lambdas: np.ndarray = _read_csv(path + 'city_traffic_lambda_table.csv')
    attribution: np.ndarray = _read_csv(path + 'city_traffic_dest_attribution_table.csv')
    distances: np.ndarray = _read_csv(path + 'city_distance_matrix_time_step.csv')

    # check size
    lts = lambdas.shape
    apts = attribution.shape
    dts = distances.shape
    assert lts[0] == apts[0] - 1 == apts[1] - 1 == dts[0] - 1 == dts[1] - 1, \
        "lambdas shape: {}\nattribution shape: {}\ndistance shape: {}".format(lts, apts, dts)

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

    # create env
    e = TFAEnvironment()

    # create nodes
    for name in lambdas.keys():
        DummyNode(name, {}, 100, [], e)
    e.build_node_state()

    lambdas, attribution, distances = _to_node_keys(e, lambdas, attribution, distances)
    e.set_distance(distances)

    # create shippers


    return e, lambdas, attribution, distances

    # create nodes

    # create dict
    # create objects


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


print("end")
