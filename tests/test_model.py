import random
from pathlib import Path

import numpy as np
import pytest

from genetic_matching.model import MatchingGA, run_experiment
from genetic_matching.preferences import load_preferences_from_file

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_fitness_sums_both_partners_preference_ranks() -> None:
    preferences = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    solver = MatchingGA(preferences, preferences, 4, 2)

    assert solver.score_solution([0, 1, 2]) == 6

    with pytest.raises(ValueError, match="permutation"):
        solver.score_solution([0, 0, 2])


def test_crossover_returns_a_complete_permutation() -> None:
    np.random.seed(8)
    random.seed(8)

    offspring = MatchingGA._crossover_chromosomes(
        np.array([0, 1, 2, 3, 4, 5]),
        np.array([5, 4, 3, 2, 1, 0]),
    )

    assert sorted(offspring.tolist()) == list(range(6))


def test_run_experiment_assigns_consecutive_attempt_seeds() -> None:
    preferences = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])

    result = run_experiment(
        preferences,
        preferences,
        population_size=6,
        num_generations=3,
        attempts=2,
        jobs=1,
        random_seed=20,
    )

    assert [attempt.seed for attempt in result.attempts] == [20, 21]
    assert sorted(result.best.solution) == [0, 1, 2]
    assert result.best.fitness == max(attempt.fitness for attempt in result.attempts)


@pytest.mark.parametrize(
    ("attempts", "jobs", "message"),
    [(0, 1, "attempts"), (1, 0, "jobs")],
)
def test_invalid_experiment_parallelism_is_rejected(attempts: int, jobs: int, message: str) -> None:
    preferences = np.array([[1, 2], [2, 1]])

    with pytest.raises(ValueError, match=message):
        run_experiment(preferences, preferences, attempts=attempts, jobs=jobs)


def test_fixed_seed_matches_recovered_algorithm_baseline() -> None:
    men, women = load_preferences_from_file(REPOSITORY_ROOT / "data/GA_input.txt")

    result = MatchingGA(
        men,
        women,
        population_size=75,
        num_generations=240,
        random_seed=2024,
    ).run_result()

    assert result.fitness == 1568
    assert result.solution == (
        8,
        14,
        4,
        3,
        0,
        5,
        6,
        7,
        18,
        12,
        10,
        1,
        22,
        13,
        15,
        25,
        16,
        21,
        17,
        19,
        2,
        11,
        9,
        23,
        24,
        20,
        26,
        27,
        28,
        29,
    )
    assert result.best_fitness[0] == 1010
    assert result.best_fitness[-1] == 1568
    assert result.average_fitness[0] == pytest.approx(930.6666666666666)
    assert result.average_fitness[-1] == pytest.approx(1520.8933333333334)
    assert result.generations == 240
