"""Deterministic result serialization and plotting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .model import AttemptResult, ExperimentResult


def write_summary(result: ExperimentResult, output_directory: str | Path) -> Path:
    """Write matching and attempt summaries as JSON."""
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "result.json"
    payload = {
        "population_size": result.population_size,
        "num_generations": result.num_generations,
        "attempts": len(result.attempts),
        "best_attempt": result.best.attempt,
        "best_seed": result.best.seed,
        "fitness": result.best.fitness,
        "matching": [{"man": man, "woman": woman} for man, woman in result.best.one_based_matching],
        "attempt_fitness": [
            {
                "attempt": attempt.attempt,
                "seed": attempt.seed,
                "fitness": attempt.fitness,
            }
            for attempt in result.attempts
        ],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_fitness_history(result: AttemptResult, output_directory: str | Path) -> Path:
    """Write the best attempt's per-generation metrics as CSV."""
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "fitness-history.csv"
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("generation", "best", "average", "worst"))
        writer.writerow((0, result.best_fitness[0], "", ""))
        for generation in range(1, result.generations + 1):
            writer.writerow(
                (
                    generation,
                    result.best_fitness[generation],
                    result.average_fitness[generation - 1],
                    result.worst_fitness[generation - 1],
                )
            )
    return output_path


def plot_fitness_metrics(
    result: AttemptResult,
    output_directory: str | Path,
    *,
    show: bool = False,
) -> tuple[Path, Path, Path]:
    """Save best, average, and worst fitness plots for one attempt."""
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    series = (
        ("best", result.best_fitness, 0),
        ("average", result.average_fitness, 1),
        ("worst", result.worst_fitness, 1),
    )
    for label, values, start_generation in series:
        figure, axes = plt.subplots()
        axes.plot(
            range(start_generation, start_generation + len(values)),
            values,
            linewidth=3,
            color="#64f20c",
        )
        axes.set_title(f"Generation vs {label.title()} Fitness")
        axes.set_xlabel("Generation")
        axes.set_ylabel("Fitness")
        output_path = destination / f"{label}.png"
        figure.savefig(output_path, bbox_inches="tight")
        paths.append(output_path)

    if show:
        plt.show()
    plt.close("all")
    return paths[0], paths[1], paths[2]
