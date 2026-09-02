import csv
import json

from genetic_matching.model import AttemptResult, ExperimentResult
from genetic_matching.reporting import (
    plot_fitness_metrics,
    write_fitness_history,
    write_summary,
)


def sample_result() -> ExperimentResult:
    attempt = AttemptResult(
        attempt=0,
        seed=7,
        solution=(1, 0),
        fitness=4,
        best_fitness=(2.0, 3.0, 4.0),
        average_fitness=(2.5, 3.5),
        worst_fitness=(1.0, 2.0),
    )
    return ExperimentResult(attempt, (attempt,), population_size=4, num_generations=2)


def test_result_files_use_stable_names_and_one_based_matches(tmp_path) -> None:
    result = sample_result()

    summary_path = write_summary(result, tmp_path)
    history_path = write_fitness_history(result.best, tmp_path)
    plot_paths = plot_fitness_metrics(result.best, tmp_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["matching"] == [{"man": 1, "woman": 2}, {"man": 2, "woman": 1}]
    with history_path.open(encoding="utf-8", newline="") as history_file:
        rows = list(csv.reader(history_file))
    assert rows == [
        ["generation", "best", "average", "worst"],
        ["0", "2.0", "", ""],
        ["1", "3.0", "2.5", "1.0"],
        ["2", "4.0", "3.5", "2.0"],
    ]
    assert [path.name for path in plot_paths] == ["best.png", "average.png", "worst.png"]
    assert all(path.is_file() for path in plot_paths)
