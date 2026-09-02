"""
Module for logic of running the genetic algorithm solver
"""
import math
import os
import random
from datetime import datetime
from math import floor
from pathlib import Path
from typing import List, Optional, Set, Tuple

import matplotlib.pyplot
import numpy as np
from pygad import GA

from src.utils import build_index_based_preferences


class FitnessMetricsGA(GA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.average_fitness: List[int] = []
        self.worst_fitness: List[int] = []


class MatchingGA:
    """
    runner for solving a matching problem using a genetic algorithm
    """
    PARENTS_MATING_RATIO: float = 0.8  # chance of random chromosome to reproduce
    ELITISM_RATIO: float = 0.2  # percentage of chromosomes of the best fitness to continue to the next generation

    MUTATION_PROBABILITY: float = 0.05  # chance of each gene in a chromosome to experience mutation

    def __init__(self, men_preferences: np.ndarray, women_preferences: np.ndarray,
                 population_size: int, num_generations: int):
        """

        :param men_preferences: value-based ranking of the men's preferences
        :param women_preferences:value-based ranking of the women's preferences
        :param population_size: size of the chromosome population
        :param num_generations: number of generations to run the genetic algorithm for
        """
        self._men_preferences = build_index_based_preferences(men_preferences)
        self._women_preferences = build_index_based_preferences(women_preferences)
        self._group_size = men_preferences.shape[0]
        # maximal theoretical value of the fitness function for the group size
        self._maximum_unhappy_matching_rating = self._group_size * (self._group_size - 1) * 2

        self._population_size = population_size
        self._num_generations = num_generations

        self._ga_instance: Optional[FitnessMetricsGA] = None
        self._mutation_probability: Optional[float] = None

    def run(self) -> FitnessMetricsGA:
        """
        Run the solver
        :return: the genetic algorithm instance result, to perform analysis
        """
        initial_matches: np.ndarray = np.array(tuple(
            np.random.permutation(np.arange(self._group_size))
            for _ in range(self._population_size)
        ))
        self._mutation_probability = self.MUTATION_PROBABILITY

        # number of parents to mate in generation
        num_parents_mating = floor(self._population_size * self.PARENTS_MATING_RATIO)
        # number of best-fitness chromosomes to copy to the next generation
        num_elitism = floor(self._population_size * self.ELITISM_RATIO)

        self._ga_instance = FitnessMetricsGA(gene_type=int,
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

    def _get_match_fitness(self, solution: np.array) -> int:
        """
        Return the fitness of a certain matching
        :param solution: matching to asses fitness on
        :return: integer of matching fitness
        """
        solution_unhappiness = 0
        for man_index, woman_index in enumerate(solution):
            solution_unhappiness += (self._women_preferences[woman_index][man_index] +
                                     self._men_preferences[man_index][woman_index])

        return self._maximum_unhappy_matching_rating - solution_unhappiness

    def _fitness_func(self, ga_instance, solution, solution_idx) -> int:
        """
        Wrapper function for returning the fitness of a matching to comply to the pygad api
        :param ga_instance: current instance of the genetic algorithm
        :param solution: chromosome to calculate fitness for
        :param solution_idx: index of the chromosome
        :return: fitness
        """
        return self._get_match_fitness(solution)

    def custom_mutation_func(self, offsprings: np.ndarray, ga_instance: FitnessMetricsGA) -> np.ndarray:
        """
        Swap mutation implementation
        :param offsprings: offsprings to mutate
        :param ga_instance: current genetic algorithm instance
        :return: offsprings after mutation
        """
        num_mutations = math.floor(self._mutation_probability * offsprings.shape[1])

        for offspring_idx in range(offsprings.shape[0]):

            for _ in range(num_mutations):
                swap_indices = np.random.choice(offsprings.shape[1], size=2, replace=True)
                offsprings[offspring_idx][swap_indices[0]], offsprings[offspring_idx][swap_indices[1]] = \
                    offsprings[offspring_idx][swap_indices[1]], offsprings[offspring_idx][swap_indices[0]]
        return offsprings

    @staticmethod
    def _crossover_chromosomes(first_parent: np.array, second_parent: np.array) -> np.array:
        """
        Generational crossover between two chromosomes
        :param first_parent: parent to crossover
        :param second_parent: parent to crossover
        :return: chromosome after crossover
        """
        if np.array_equal(first_parent, second_parent):  # the same parent was chosen
            return first_parent

        offspring: np.array = np.empty_like(first_parent)
        offspring[:] = -1
        values_in_offspring: Set[int] = set()
        women_indices: Set[int] = set(range(first_parent.shape[0]))

        # randomly walk on the parent's matchings
        for first_man_index in np.random.permutation(first_parent.shape[0]):
            first_woman_index = first_parent[first_man_index]
            # randomly choose a value for a gene from father or mother
            man_index, woman_index = random.choice(
                [(first_man_index, first_woman_index), (first_man_index, second_parent[first_man_index])]
            )
            # a woman is not yet matched to a man - to prevent "double booking"
            if woman_index not in values_in_offspring:
                offspring[man_index] = woman_index
                values_in_offspring.add(woman_index)

        # permute over unmatched women over the unmatched men
        unmatched_women = women_indices - values_in_offspring
        offspring[offspring == -1] = np.random.permutation(np.array(list(unmatched_women)))
        return offspring

    def _crossover_func(self, parents: np.ndarray, offspring_size: Tuple[int, int],
                        ga_instance: FitnessMetricsGA) -> np.ndarray:
        """
        Crossover function for the pygad api
        :param parents: chosen parents to perform crossover on
        :param offspring_size: size of the expected offsprings
        :param ga_instance: current instance of the genetic algorithm runner
        :return: offsprings after crossover
        """
        offsprings = np.zeros(offspring_size)
        parents_fitness = ga_instance.last_generation_fitness[ga_instance.last_generation_parents_indices]
        total_fitness = np.sum(parents_fitness)

        for offspring_row in range(offspring_size[0]):
            chosen_parents_rows = np.random.choice(parents.shape[0], 2,
                                                   p=parents_fitness / total_fitness)
            offsprings[offspring_row] = self._crossover_chromosomes(parents[chosen_parents_rows[0]],
                                                                    parents[chosen_parents_rows[1]])

        return offsprings

    def _keep_generation_metrics(self, ga_instance: FitnessMetricsGA):
        ga_instance.average_fitness.append(np.mean(self._ga_instance.last_generation_fitness))
        ga_instance.worst_fitness.append(np.min(self._ga_instance.last_generation_fitness))


def plot_fitness_metrics(ga_instance: FitnessMetricsGA, plot_save_path: Path):
    """
    Plots the metrics on the FitnessMetricsGA (best, worst and mean fitness in each generation)
    :param ga_instance: FitnessMetricsGA instance
    :param plot_save_path: path to save the plot tp
    """
    os.makedirs(plot_save_path, exist_ok=True)
    # plot metrics on the generations
    best_generations_fitness = ga_instance.best_solutions_fitness
    average_generations_fitness = ga_instance.average_fitness
    worst_generations_fitness = ga_instance.worst_fitness

    best_fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(best_generations_fitness,
                           linewidth=3,
                           color="#64f20c")
    matplotlib.pyplot.title("Generation vs Best Fitness", fontsize=14)
    matplotlib.pyplot.xlabel("Generation", fontsize=14)
    matplotlib.pyplot.ylabel("Fitness", fontsize=14)
    matplotlib.pyplot.savefig(fname=plot_save_path /
                                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_best_{ga_instance.best_solution()[1]}",
                              bbox_inches='tight')

    average_fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(average_generations_fitness,
                           linewidth=3,
                           color="#64f20c")
    matplotlib.pyplot.title("Generation vs Average Fitness", fontsize=14)
    matplotlib.pyplot.xlabel("Generation", fontsize=14)
    matplotlib.pyplot.ylabel("Fitness", fontsize=14)
    matplotlib.pyplot.savefig(fname=plot_save_path /
                                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_mean_{ga_instance.best_solution()[1]}",
                              bbox_inches='tight')

    worst_fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(worst_generations_fitness,
                           linewidth=3,
                           color="#64f20c")
    matplotlib.pyplot.title("Generation vs Worst Fitness", fontsize=14)
    matplotlib.pyplot.xlabel("Generation", fontsize=14)
    matplotlib.pyplot.ylabel("Fitness", fontsize=14)
    matplotlib.pyplot.savefig(fname=plot_save_path /
                                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_worst_{ga_instance.best_solution()[1]}",
                              bbox_inches='tight')

    matplotlib.pyplot.show(block=True)
