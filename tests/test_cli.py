import json

from genetic_matching.cli import cli


def test_cli_runs_a_small_deterministic_experiment(monkeypatch, tmp_path, capsys) -> None:
    input_path = tmp_path / "preferences.txt"
    input_path.write_text("1 2\n2 1\n2 1\n1 2\n", encoding="utf-8")
    output_directory = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "genetic-matching",
            str(input_path),
            "--population-size",
            "4",
            "--generations",
            "2",
            "--attempts",
            "1",
            "--jobs",
            "1",
            "--seed",
            "7",
            "--output-dir",
            str(output_directory),
        ],
    )

    cli()

    payload = json.loads((output_directory / "result.json").read_text(encoding="utf-8"))
    assert payload["attempts"] == 1
    assert payload["best_seed"] == 7
    assert "matching (<man> - <woman>)" in capsys.readouterr().out
