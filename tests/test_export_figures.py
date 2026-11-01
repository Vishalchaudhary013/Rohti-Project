from __future__ import annotations

from scripts import export_figures
from src import visuals
from tests.conftest import make_frame


def test_export_cli(tmp_path, capsys):
    make_frame(40, 1).to_csv(tmp_path / "student-mat.csv", sep=";", index=False)
    make_frame(50, 2).to_csv(tmp_path / "student-por.csv", sep=";", index=False)
    out = tmp_path / "figs"
    code = export_figures.main(
        [
            "--input",
            str(tmp_path / "student-mat.csv"),
            str(tmp_path / "student-por.csv"),
            "--out",
            str(out),
            "--dpi",
            "40",
            "--exclude-g3-zero",
        ]
    )
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(visuals.FIGURE_CATALOG)
    assert "Exported 12 figures" in capsys.readouterr().out


def test_export_cli_errors(tmp_path, capsys):
    assert export_figures.main(["--input", str(tmp_path / "missing.csv")]) == 1
    make_frame(5, 1).drop(columns=["G3"]).to_csv(tmp_path / "bad.csv", index=False)
    assert export_figures.main(["--input", str(tmp_path / "bad.csv")]) == 1
    assert "no file passed" in capsys.readouterr().err
