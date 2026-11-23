from __future__ import annotations

from scripts import make_sample_data
from src import config, ingestion


def test_generated_files_validate_and_are_deterministic(tmp_path):
    assert make_sample_data.main(["--out", str(tmp_path)]) == 0
    paths = [tmp_path / name for name in make_sample_data.FILE_NAMES.values()]
    result = ingestion.load_files(paths)
    assert all(r.is_valid and r.count("error") == 0 for r in result.reports.values())
    counts = result.dataset[config.SUBJECT_COL].value_counts().to_dict()
    assert counts == {"por": 649, "mat": 395}
    assert result.dataset[config.BOTH_SUBJECTS_FLAG_COL].any()
    again = make_sample_data.generate()
    assert again["mat"].equals(make_sample_data.generate()["mat"])


def test_custom_seed_prefix_and_sizes(tmp_path):
    args = [
        "--out",
        str(tmp_path),
        "--seed",
        "7",
        "--prefix",
        "random",
        "--mat-rows",
        "50",
        "--por-rows",
        "80",
    ]
    assert make_sample_data.main(args) == 0
    result = ingestion.load_files(
        [tmp_path / "random-student-mat.csv", tmp_path / "random-student-por.csv"]
    )
    assert result.dataset[config.SUBJECT_COL].value_counts().to_dict() == {"por": 80, "mat": 50}
    assert not make_sample_data.generate(7)["mat"].equals(make_sample_data.generate(8)["mat"])
