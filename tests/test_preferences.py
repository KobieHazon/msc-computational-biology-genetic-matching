from pathlib import Path

import numpy as np
import pytest

from genetic_matching.preferences import (
    build_index_based_preferences,
    load_preferences_from_file,
)

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_supplied_input_contains_two_valid_30_by_30_matrices() -> None:
    men, women = load_preferences_from_file(REPOSITORY_ROOT / "data/GA_input.txt")

    assert men.shape == (30, 30)
    assert women.shape == (30, 30)
    expected = list(range(1, 31))
    assert all(sorted(row.tolist()) == expected for row in men)
    assert all(sorted(row.tolist()) == expected for row in women)


def test_rank_conversion_does_not_mutate_the_input() -> None:
    preferences = np.array([[3, 1, 2], [2, 3, 1], [1, 2, 3]])
    original = preferences.copy()

    ranks = build_index_based_preferences(preferences)

    np.testing.assert_array_equal(preferences, original)
    np.testing.assert_array_equal(ranks, [[1, 2, 0], [2, 0, 1], [0, 1, 2]])


@pytest.mark.parametrize(
    "contents",
    [
        "1 2\n2 1\n1 2\n",
        "1 1\n2 1\n1 2\n2 1\n",
        "1 2 3\n2 1 3\n1 3 2\n2 1 3\n",
    ],
)
def test_invalid_input_is_rejected(tmp_path, contents: str) -> None:
    input_path = tmp_path / "invalid.txt"
    input_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        load_preferences_from_file(input_path)
