"""Genetic-algorithm model for two-sided preference matching."""

from __future__ import annotations

import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import floor

import numpy as np
import numpy.typing as npt
from pygad import GA

from .preferences import PreferenceMatrix, build_index_based_preferences


class FitnessMetricsGA(GA):
    """PyGAD runner extended with mean and worst generation histories."""

    def __init__(self, *args: object, **kwargs: object):
        super().__init__(*args, **kwargs)
        self.average_fitness: list[float] = []
        self.worst_fitness: list[float] = []


@dataclass(frozen=True)
class AttemptResult:
    """Serializable result from one genetic-algorithm attempt."""

    attempt: int
    seed: int | None
    solution: tuple[int, ...]
    fitness: int
    best_fitness: tuple[float, ...]
    average_fitness: tuple[float, ...]
    worst_fitness: tuple[float, ...]

    @property
    def generations(self) -> int:
        return len(self.average_fitness)

    @property
    def one_based_matching(self) -> tuple[tuple[int, int], ...]:
        return tuple((man + 1, woman + 1) for man, woman in enumerate(self.solution))


@dataclass(frozen=True)
class ExperimentResult:
    """Best result and attempt-level summaries for an experiment."""

    best: AttemptResult
    attempts: tuple[AttemptResult, ...]
    population_size: int
    num_generations: int


class MatchingGA:
    """Optimize a one-to-one matching with custom permutation operators."""

    PARENTS_MATING_RATIO = 0.8
    ELITISM_RATIO = 0.2
    MUTATION_PROBABILITY = 0.05

    def __init__(
        self,
        men_preferences: PreferenceMatrix,
        women_preferences: PreferenceMatrix,
        population_size: int,
        num_generations: int,
        *,
        random_seed: int | None = None,
    ):
        if men_preferences.shape != women_preferences.shape:
            raise ValueError("men and women preference matrices must have equal shapes")
        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        if num_generations < 1:
            raise ValueError("num_generations must be positive")

        self._men_preferences = build_index_based_preferences(men_preferences)
        self._women_preferences = build_index_based_preferences(women_preferences)
        self._group_size = men_preferences.shape[0]
        self._maximum_unhappy_matching_rating = self._group_size * (self._group_size - 1) * 2
        self._population_size = population_size
        self._num_generations = num_generations
        self._random_seed = random_seed
        self._mutation_probability = self.MUTATION_PROBABILITY
        self._ga_instance: FitnessMetricsGA | None = None

    def run(self) -> FitnessMetricsGA:
        """Run one genetic-algorithm attempt."""
        if self._random_seed is not None:
            random.seed(self._random_seed)
            np.random.seed(self._random_seed)

        initial_matches = np.array(
            [np.random.permutation(self._group_size) for _ in range(self._population_size)],
            dtype=int,
        )
        num_parents_mating = max(1, floor(self._population_size * self.PARENTS_MATING_RATIO))
        num_elitism = floor(self._population_size * self.ELITISM_RATIO)

        self._ga_instance = FitnessMetricsGA(
            gene_type=int,
            initial_population=initial_matches,
            num_generations=self._num_generations,
            fitness_func=self._fitness_func,
            mutation_type=self.custom_mutation_func,
            crossover_type=self._crossover_func,
            num_parents_mating=num_parents_mating,
            parent_selection_type="tournament",
            keep_elitism=num_elitism,
            on_generation=self._keep_generation_metrics,
            suppress_warnings=True,
        )
        self._ga_instance.run()
        return self._ga_instance

    def run_result(self, attempt: int = 0) -> AttemptResult:
        """Run and reduce the PyGAD object to a portable result."""
        ga_instance = self.run()
        solution, fitness, _solution_index = ga_instance.best_solution()
        return AttemptResult(
            attempt=attempt,
            seed=self._random_seed,
            solution=tuple(int(value) for value in solution),
            fitness=int(fitness),
            best_fitness=tuple(float(value) for value in ga_instance.best_solutions_fitness),
            average_fitness=tuple(ga_instance.average_fitness),
            worst_fitness=tuple(ga_instance.worst_fitness),
        )

    def _get_match_fitness(self, solution: npt.NDArray[np.int_]) -> int:
        men = np.arange(self._group_size)
        women = np.asarray(solution, dtype=int)
        solution_unhappiness = int(
            np.sum(self._women_preferences[women, men] + self._men_preferences[men, women])
        )
        return self._maximum_unhappy_matching_rating - solution_unhappiness

    def score_solution(self, solution: npt.ArrayLike) -> int:
        """Validate and score one zero-based one-to-one matching."""
        candidate = np.asarray(solution, dtype=int)
        if candidate.shape != (self._group_size,) or not np.array_equal(
            np.sort(candidate), np.arange(self._group_size)
        ):
            raise ValueError("solution must be a permutation of all group indices")
        return self._get_match_fitness(candidate)

    def _fitness_func(
        self,
        _ga_instance: GA,
        solution: npt.NDArray[np.int_],
        _solution_index: int,
    ) -> int:
        return self._get_match_fitness(solution)

    def custom_mutation_func(
        self,
        offspring: npt.NDArray[np.int_],
        _ga_instance: FitnessMetricsGA,
    ) -> npt.NDArray[np.int_]:
        """Apply the swap mutation while preserving permutations."""
        num_mutations = math.floor(self._mutation_probability * offspring.shape[1])
        for offspring_index in range(offspring.shape[0]):
            for _ in range(num_mutations):
                first, second = np.random.choice(offspring.shape[1], size=2, replace=True)
                row = offspring[offspring_index]
                row[first], row[second] = row[second], row[first]
        return offspring

    @staticmethod
    def _crossover_chromosomes(
        first_parent: npt.NDArray[np.int_],
        second_parent: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.int_]:
        """Combine two permutations without duplicating a match."""
        if np.array_equal(first_parent, second_parent):
            return first_parent.copy()

        offspring = np.full_like(first_parent, -1)
        values_in_offspring: set[int] = set()
        women_indices = set(range(first_parent.shape[0]))
        for man_index in np.random.permutation(first_parent.shape[0]):
            selected_man, woman_index = random.choice(
                [
                    (man_index, first_parent[man_index]),
                    (man_index, second_parent[man_index]),
                ]
            )
            if int(woman_index) not in values_in_offspring:
                offspring[selected_man] = woman_index
                values_in_offspring.add(int(woman_index))

        unmatched_women = women_indices - values_in_offspring
        offspring[offspring == -1] = np.random.permutation(np.array(list(unmatched_women)))
        return offspring

    def _crossover_func(
        self,
        parents: npt.NDArray[np.int_],
        offspring_size: tuple[int, int],
        ga_instance: FitnessMetricsGA,
    ) -> npt.NDArray[np.float64]:
        offspring = np.zeros(offspring_size)
        parent_indices = ga_instance.last_generation_parents_indices
        parents_fitness = ga_instance.last_generation_fitness[parent_indices]
        total_fitness = float(np.sum(parents_fitness))
        probabilities = parents_fitness / total_fitness if total_fitness > 0 else None

        for row in range(offspring_size[0]):
            chosen = np.random.choice(
                parents.shape[0],
                size=2,
                replace=True,
                p=probabilities,
            )
            offspring[row] = self._crossover_chromosomes(parents[chosen[0]], parents[chosen[1]])
        return offspring

    def _keep_generation_metrics(self, ga_instance: FitnessMetricsGA) -> None:
        assert self._ga_instance is not None
        ga_instance.average_fitness.append(
            float(np.mean(self._ga_instance.last_generation_fitness))
        )
        ga_instance.worst_fitness.append(float(np.min(self._ga_instance.last_generation_fitness)))


def run_attempt(
    men_preferences: PreferenceMatrix,
    women_preferences: PreferenceMatrix,
    population_size: int,
    num_generations: int,
    seed: int | None,
    attempt: int,
) -> AttemptResult:
    """Run one process-safe attempt."""
    return MatchingGA(
        men_preferences,
        women_preferences,
        population_size,
        num_generations,
        random_seed=seed,
    ).run_result(attempt)


def run_experiment(
    men_preferences: PreferenceMatrix,
    women_preferences: PreferenceMatrix,
    *,
    population_size: int = 75,
    num_generations: int = 240,
    attempts: int = 100,
    jobs: int | None = None,
    random_seed: int | None = None,
) -> ExperimentResult:
    """Run independent attempts and return the highest-fitness result."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if jobs is not None and jobs < 1:
        raise ValueError("jobs must be positive")
    available_cpus = os.cpu_count() or 1
    worker_count = min(attempts, jobs if jobs is not None else available_cpus)

    seeds = [
        random_seed + attempt if random_seed is not None else None for attempt in range(attempts)
    ]
    arguments = [
        (
            men_preferences,
            women_preferences,
            population_size,
            num_generations,
            seeds[attempt],
            attempt,
        )
        for attempt in range(attempts)
    ]

    if worker_count == 1:
        results = tuple(run_attempt(*argument) for argument in arguments)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(run_attempt, *argument) for argument in arguments]
            results = tuple(future.result() for future in futures)

    best = max(results, key=lambda result: result.fitness)
    return ExperimentResult(best, results, population_size, num_generations)
