"""Command-line interface for the genetic-matching solver."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model import run_experiment
from .preferences import load_preferences_from_file
from .reporting import plot_fitness_metrics, write_fitness_history, write_summary

DEFAULT_EVALUATION_BUDGET = 18_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=Path("data/GA_input.txt"),
    )
    parser.add_argument("--population-size", type=int, default=75)
    parser.add_argument(
        "--generations",
        type=int,
        help="defaults to 18000 divided by the population size",
    )
    parser.add_argument("--attempts", type=int, default=100)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("run-results"))
    parser.add_argument("--show", action="store_true")
    return parser


def cli() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.population_size < 2:
        parser.error("--population-size must be at least 2")
    generations = (
        arguments.generations
        if arguments.generations is not None
        else DEFAULT_EVALUATION_BUDGET // arguments.population_size
    )
    if generations < 1:
        parser.error("--generations must be positive")
    men_preferences, women_preferences = load_preferences_from_file(arguments.input_file)
    result = run_experiment(
        men_preferences,
        women_preferences,
        population_size=arguments.population_size,
        num_generations=generations,
        attempts=arguments.attempts,
        jobs=arguments.jobs,
        random_seed=arguments.seed,
    )
    write_summary(result, arguments.output_dir)
    write_fitness_history(result.best, arguments.output_dir)
    plot_fitness_metrics(result.best, arguments.output_dir, show=arguments.show)

    print("matching (<man> - <woman>) =")
    print("\n".join(f"{man} - {woman}" for man, woman in result.best.one_based_matching))
    print(f"fitness = {result.best.fitness}")
    print(f"best attempt = {result.best.attempt}, seed = {result.best.seed}")


if __name__ == "__main__":
    cli()
