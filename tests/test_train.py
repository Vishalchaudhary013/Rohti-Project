from __future__ import annotations

import json

from scripts import train
from tests.conftest import make_frame
from tests.test_modeling import _learnable


def test_train_cli_writes_reports_and_model(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    df = _learnable(120, 3)
    cols = list(config.REQUIRED_COLUMNS)
    df[df.subject == "mat"][cols].to_csv(tmp_path / "student-mat.csv", sep=";", index=False)
    df[df.subject == "por"][cols].to_csv(tmp_path / "student-por.csv", sep=";", index=False)
    out = tmp_path / "metrics"
    code = train.main(
        [
            "--input",
            str(tmp_path / "student-mat.csv"),
            str(tmp_path / "student-por.csv"),
            "--out",
            str(out),
            "--save",
            "logistic_regression",
            "--setting",
            "B",
            "--balanced",
        ]
    )
    assert code == 0
    report = json.loads((out / "metrics.json").read_text())
    assert report["n_rows"] == 120 and len(report["results"]) == 8
    assert "Majority-class baseline" in (out / "metrics.md").read_text()
    assert (tmp_path / "models" / "latest.json").exists()


def test_train_cli_errors(tmp_path, capsys):
    assert train.main(["--input", str(tmp_path / "nope.csv")]) == 1
    small = make_frame(6, 1)
    small["G3"] = 15
    small.to_csv(tmp_path / "s.csv", index=False)
    assert train.main(["--input", str(tmp_path / "s.csv"), "--out", str(tmp_path)]) == 1
    assert "at least" in capsys.readouterr().err
