"""Preference-matrix parsing and ranking conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

PreferenceMatrix = npt.NDArray[np.int_]


def validate_preferences(preferences: PreferenceMatrix) -> None:
    """Validate a square matrix whose rows are one-based preference orders."""
    if preferences.ndim != 2 or preferences.shape[0] != preferences.shape[1]:
        raise ValueError("each preference matrix must be square")

    group_size = preferences.shape[0]
    expected = np.arange(1, group_size + 1)
    if any(not np.array_equal(np.sort(row), expected) for row in preferences):
        raise ValueError(f"each preference row must be a permutation of 1 through {group_size}")


def load_preferences_from_file(
    input_file_path: str | Path,
) -> tuple[PreferenceMatrix, PreferenceMatrix]:
    """Load equal-size men's and women's preference matrices from one file."""
    source = Path(input_file_path)
    try:
        preferences = np.loadtxt(source, dtype=int)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not load preferences from {source}") from error

    if preferences.ndim != 2 or preferences.shape[0] % 2:
        raise ValueError("the input must contain two equally sized preference matrices")

    group_size = preferences.shape[0] // 2
    if preferences.shape[1] != group_size:
        raise ValueError("the input must contain 2n rows with n preference values per row")

    men = preferences[:group_size].copy()
    women = preferences[group_size:].copy()
    validate_preferences(men)
    validate_preferences(women)
    return men, women


def build_index_based_preferences(
    preferences_grid: PreferenceMatrix,
) -> PreferenceMatrix:
    """Convert one-based preference orders to zero-based rank lookups."""
    preferences = np.asarray(preferences_grid, dtype=int)
    validate_preferences(preferences)

    zero_based = preferences.copy() - 1
    ranks = np.empty_like(zero_based)
    group_size = preferences.shape[0]
    ranks[np.arange(group_size)[:, None], zero_based] = np.arange(group_size)
    return ranks
