# Genetic Preference Matching

A 2024 CS MSc Computational Biology project that uses a genetic algorithm to optimize one-to-one matches between two groups of 30 participants. Each side supplies a complete preference ordering; custom crossover and swap-mutation operators preserve valid permutations while PyGAD evolves candidates toward lower combined dissatisfaction.

## Tech Stack

- Python 3.10 or newer
- NumPy for preference matrices, ranking conversion, and permutation operations
- PyGAD 3.3.1 for genetic-algorithm lifecycle and parent selection
- multiprocessing through `ProcessPoolExecutor` for independent attempts
- Matplotlib for best, mean, and worst fitness plots
- pytest, Ruff, and uv for validation and reproducibility

## Fitness Model

Input rows rank candidates from most to least preferred using one-based identifiers. The implementation converts each row to a zero-based rank lookup and sums both partners' ranks for every proposed pair. Fitness is:

```text
2 * group_size * (group_size - 1) - total_pair_dissatisfaction
```

For the supplied 30-by-30 input, the theoretical maximum is `1740`; larger fitness is better. This is an optimization of mutual preference rank, not an implementation of the stable-marriage algorithm.

## Setup

```bash
git clone https://github.com/KobieHazon/msc-computational-biology-genetic-matching.git
cd msc-computational-biology-genetic-matching
uv sync --dev
```

## Usage

Run one deterministic attempt using the recovered population and generation settings:

```bash
uv run genetic-matching data/GA_input.txt --attempts 1 --jobs 1 --seed 2024
```

This regression case finishes with fitness `1568`. It is a regression-test seed applied to the unmodified recovered algorithm, not a recovered seed from the historical experiments.

Run the full 100-attempt search:

```bash
uv run genetic-matching data/GA_input.txt --seed 2024
```

The validated run selects attempt 76, seed `2100`, with fitness `1610`; two complete executions produced identical JSON, CSV, and PNG outputs. The defaults use a population of 75 for 240 generations, preserving the exercise's `18,000` population-generation budget. Independent attempts use consecutive seeds and no more than the available CPU count. Use `--population-size`, `--generations`, `--attempts`, and `--jobs` to control the workload.

Each run writes `result.json`, `fitness-history.csv`, and three plots under `run-results/`. Matching identifiers printed by the maintained command are one-based, consistent with the input format. Pass `--show` to display the generated figures interactively.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The suite checks input validation, non-mutating rank conversion, fitness semantics, permutation-preserving operators, deterministic serialization, the command line, and the complete fixed-seed regression.

## Repository Structure

- `assignment/`: supplied exercise brief, converted to a PDF
- `data/GA_input.txt`: supplied 30-by-30 preference input; the recovered local filename was normalized to the name used in the brief
- `src/genetic_matching/`: solver, validated parser, reporting, and command-line interface
- `results/historical-plots/`: all 33 distinct recovered experiment plots and a provenance note
- `tests/`: focused behavior and reproducibility checks

## Implementation notes

No separate report was recovered. The 33 plots are therefore retained as the available experiment record. Several historical runs reached fitness `1611`, but their seeds and complete launch parameters were not recorded, so the README does not claim that they can be reproduced exactly.

The maintained version adds a portable CLI, deterministic per-attempt seeds and outputs, bounded worker selection, input validation, non-mutating conversion, current packaging, and regression tests. PyGAD remains pinned to `3.3.1`: changing the offspring-array conversion behavior or PyGAD generation can change the stochastic trajectory even with the same seed.

## Authorship

Solution and generated experiment plots by Kobie Hazon and Daniel Ben Zion. The exercise brief and input are identified as supplied material solely to distinguish them from the coauthored work; this label does not assert ownership by a university or any other institution.

No repository-wide license is declared because the repository combines original work with supplied material whose reuse terms were not recorded.
