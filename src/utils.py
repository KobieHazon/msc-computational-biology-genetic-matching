"""
Utils for the matching problem genetic algorithm solver
"""
from pathlib import Path
from typing import Tuple

import numpy as np


def load_preferences_from_file(input_file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load preferences matrix from the file
    :param input_file_path: path for file where the first half contains order-based ranking of the women and the second half for the men
    :return: tuple of 2 matrices, first for men and second for women
    """
    try:
        with open(input_file_path, "r") as input_file:
            line_count = len(input_file.readlines())
            input_file.seek(0)
            men_preferences_grid = np.loadtxt(input_file, max_rows=line_count // 2, dtype=int)
            # continues from the previous line cursor
            women_preferences_grid = np.loadtxt(input_file, max_rows=line_count // 2, dtype=int)
    except Exception as e:
        raise ValueError(f"could not load preferences from {input_file_path}") from e
    return men_preferences_grid, women_preferences_grid


def build_index_based_preferences(preferences_grid: np.ndarray) -> np.ndarray:
    """
    Converts the format of the preferences grid for the matching-problem genetic solver
    :param preferences_grid: preferences file as in the input file - order-based ranking
    :return: matrix for the preferences where for each group member is preference is value-based ranked
    """
    index_preferences = np.zeros_like(preferences_grid)
    preferences_grid -= 1
    for preference_index, preference in enumerate(preferences_grid):
        for column, preference_value in enumerate(preference):
            index_preferences[preference_index, preference_value] = column

    return index_preferences
