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
