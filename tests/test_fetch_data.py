from __future__ import annotations

import io
import urllib.error
import zipfile

import pytest

from scripts import fetch_data
from src import config
from tests.conftest import make_frame


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_extract_targets_from_nested_zip():
    inner = _zip({"student-mat.csv": b"m", "student-por.csv": b"p", "student.txt": b"t"})
    outer = _zip({"student.zip": inner, "../evil/student-mat.csv.bak": b"x"})
    found = fetch_data.extract_targets(outer, ("student-mat.csv", "student-por.csv"))
    assert found == {"student-mat.csv": b"m", "student-por.csv": b"p"}


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "PROCESSED_FILE", tmp_path / "processed" / "c.csv")
    targets = (raw / "student-mat.csv", raw / "student-por.csv")
    monkeypatch.setattr(fetch_data, "TARGET_FILES", targets)
    monkeypatch.setattr(config, "ALL_DIRS", (raw,))
    return raw


def test_download_failure_prints_instructions(raw_dir, monkeypatch, capsys):
    def boom(*_a, **_k):
        raise urllib.error.URLError("blocked")

    monkeypatch.setattr(fetch_data, "download", boom)
    assert fetch_data.main([]) == 1
    err = capsys.readouterr().err
    assert config.DATASET_PAGE_URL in err
    assert "student-mat.csv" in err


def test_full_run_with_mocked_download(raw_dir, monkeypatch, capsys):
    mat = make_frame(20, 1).to_csv(sep=";", index=False).encode()
    por = make_frame(30, 2).to_csv(sep=";", index=False).encode()
    archive = _zip({"student.zip": _zip({"student-mat.csv": mat, "student-por.csv": por})})
    monkeypatch.setattr(fetch_data, "download", lambda *a, **k: archive)
    assert fetch_data.main([]) == 0
    out = capsys.readouterr().out
    assert "DATA-QUALITY SUMMARY" in out
    assert "WARNING: mat has 20 rows" in out
    assert (raw_dir / "student-mat.csv").exists()
    assert config.PROCESSED_FILE.exists()


def test_archive_missing_members(raw_dir, monkeypatch, capsys):
    monkeypatch.setattr(fetch_data, "download", lambda *a, **k: _zip({"readme.txt": b"x"}))
    assert fetch_data.main([]) == 1
    assert "did not contain" in capsys.readouterr().err


def test_offline_without_files(raw_dir, monkeypatch, capsys):
    assert fetch_data.main(["--offline"]) == 1
    assert "Missing raw file" in capsys.readouterr().err
