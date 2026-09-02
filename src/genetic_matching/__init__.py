"""Genetic matching over two-sided ranked preferences."""

from .model import AttemptResult, ExperimentResult, MatchingGA, run_experiment
from .preferences import load_preferences_from_file

__all__ = [
    "AttemptResult",
    "ExperimentResult",
    "MatchingGA",
    "load_preferences_from_file",
    "run_experiment",
]
