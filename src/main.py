"""
Main module for running matching problem genetic algorithm solver
"""
import multiprocessing
import os
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
from pygad import GA
from tqdm import tqdm

from src.matching_ga import FitnessMetricsGA, MatchingGA, plot_fitness_metrics
from src.utils import load_preferences_from_file

INPUT_FILE_PATH: Path = Path(r"../input.txt")
POPULATION_SIZE: int = 75
MAX_GENERATIONS_BY_POPULATION: int = 18000  # generation multiplied by population - computational limit

SIMULATION_ATTEMPTS: int = 100  # number of times to run the genetic algorithm - to increase chances of global maxima

FITNESS_PLOT_SAVE_PATH: Path = Path(r"../matching-ga-fitness-plots/")


def run_simulation(man_preferences: np.ndarray, women_preferences: np.ndarray,
                   num_generations: int, attempt: int) -> GA:
    """
    Function to run the simulation, used by the multiprocessing worker pool
    :param man_preferences: matrix containing men preferences on the women
    :param women_preferences: matrix containing women preferences on the men
    :param num_generations: number of generations to run the simulation for
    :param attempt: the number of the current run attempt
    :return: GA instance after running the matching-ga
    """
    matching_ga = MatchingGA(man_preferences, women_preferences, POPULATION_SIZE, num_generations)
    ga_instance_result = matching_ga.run()
    return ga_instance_result


def main(input_file_path: Path):
    """
    Main flow for the matching-ga solver
    :param input_file_path: path of the men-women preferences
    """
    num_generations = MAX_GENERATIONS_BY_POPULATION // POPULATION_SIZE
    man_preferences, women_preferences = load_preferences_from_file(input_file_path)

    print(f"starting genetic algorithm:")
    mp_run_simulation = partial(run_simulation, man_preferences, women_preferences, num_generations)
    simulation_results: List[FitnessMetricsGA] = []
    with multiprocessing.Pool(processes=os.cpu_count() * 2) as pool:
        with tqdm(total=SIMULATION_ATTEMPTS) as pbar:
            for result in pool.imap_unordered(mp_run_simulation, range(SIMULATION_ATTEMPTS)):
                simulation_results.append(result)
                pbar.update()

    best_ga_result = max(simulation_results, key=lambda x: x.best_solution()[1])
    print(f"best solution:\n"
          f"matching (<man> - <woman>) = \n"
          + "\n".join(f"{man} - {woman} " for man, woman in enumerate(best_ga_result.best_solution()[0])) +
          f"\nfitness = {best_ga_result.best_solution()[1]}\n")
    # print(f"average solution fitness: "
    #       f"{sum(result.best_solution()[1] for result in simulation_results) / len(simulation_results)}")

    plot_fitness_metrics(best_ga_result, FITNESS_PLOT_SAVE_PATH)


if __name__ == "__main__":
    main(INPUT_FILE_PATH)
