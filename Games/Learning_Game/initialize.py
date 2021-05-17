"""
This files defines a few funciton to unitialize the variables in exploitation_driver.py and training_driver.py
"""

import csv
import numpy as np
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from Mechanics.Environment.tfa_environment import TFAEnvironment


def load_env() -> 'TFAEnvironment':
    path = ""
    lambdas_table: np.ndarray = _read_csv(path + 'city_traffic_lambda_table.csv')
    attribution_probabilities_table: np.ndarray = _read_csv(path + 'city_traffic_dest_attribution_table.csv')
    distances_table: np.ndarray = _read_csv(path + 'city_distance_matrix_time_step.csv')

    # check size


    # return lambdas_table, attribution_probabilities_table, distances_table
    # check size
    # check keys
    # convert
    # create dict
    # create objects


def _read_csv(file_path: str) -> np.ndarray:
    """Return a List with all the values"""
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        final: List = []
        for line in reader:
            final.append(line)
    # return np.array(final)
    return final

print("end")
