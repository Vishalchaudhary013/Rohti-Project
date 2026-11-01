from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src import config, ingestion
from tests.conftest import make_frame


# --------------------------------------------------------------------- reading
def test_read_semicolon_csv_with_quotes(frame):
    raw = frame.to_csv(sep=";", index=False, quoting=2).encode()  # QUOTE_NONNUMERIC
    df = ingestion.read_table(raw, "student-mat.csv")
    assert list(df.columns) == list(frame.columns)
    assert len(df) == len(frame)


def test_read_comma_csv_and_path(tmp_path, frame):
    path = tmp_path / "x.csv"
    frame.to_csv(path, index=False)
    df = ingestion.read_table(path)
    pd.testing.assert_frame_equal(df, frame, check_dtype=False)


def test_read_xlsx(frame):
    buf = io.BytesIO()
    frame.to_excel(buf, index=False)
    buf.seek(0)
    buf.name = "upload.xlsx"
    df = ingestion.read_table(buf)
    assert df.shape == frame.shape


def test_read_rejects_unknown_suffix():
    with pytest.raises(ValueError, match="Unsupported"):
        ingestion.read_table(b"a,b", "data.json")


def test_read_reports_parse_errors():
    with pytest.raises(ValueError, match="Could not parse"):
        ingestion.read_table(b"\xff\xfe\x00garbage", "bad.csv")


def test_checksum_matches_for_bytes_and_path(tmp_path):
    p = tmp_path / "f.csv"
    p.write_bytes(b"abc")
    assert ingestion.file_checksum(p) == ingestion.file_checksum(b"abc")
    assert len(ingestion.file_checksum(b"abc")) == 64


# ------------------------------------------------------------------ validation
def test_valid_frame_has_no_errors(frame):
    frame["G3"] = frame["G3"].clip(lower=1)  # avoid the G3 == 0 warning
    rep = ingestion.validate(frame, "ok")
    assert rep.is_valid
    assert rep.count("error") == 0
    assert rep.n_invalid_rows == 0


def test_missing_required_column_is_table_error(frame):
    rep = ingestion.validate(frame.drop(columns=["G2"]), "x")
    assert not rep.is_valid
    assert any(i.check == "required_column" and i.column == "G2" for i in rep.issues)
    with pytest.raises(ValueError):
        ingestion.clean(frame.drop(columns=["G2"]), rep)


def test_empty_table():
    rep = ingestion.validate(pd.DataFrame(), "empty")
    assert not rep.is_valid
    assert rep.issues[0].check == "empty"


@pytest.mark.parametrize(
    ("col", "value", "check"),
    [
        ("G3", 21, "range"),
        ("G1", -1, "range"),
        ("age", 30, "range"),
        ("absences", "lots", "dtype"),
        ("studytime", 2.5, "integer"),
        ("school", "XX", "domain"),
        ("higher", None, "missing"),
    ],
)
def test_row_level_errors_are_reported_and_dropped(frame, col, value, check):
    frame[col] = frame[col].astype(object)
    frame.loc[3, col] = value
    rep = ingestion.validate(frame, "x")
    assert rep.is_valid  # row errors do not reject the table
    assert any(i.check == check and i.column == col and i.n_rows == 1 for i in rep.issues)
    cleaned = ingestion.clean(frame, rep)
    assert len(cleaned) == len(frame) - 1
    assert cleaned[config.NUMERIC_COLUMNS[0]].dtype == np.int64


def test_extra_column_info_and_whitespace_trim(frame):
    frame["notes"] = "x"
    frame["school"] = " " + frame["school"] + " "
    rep = ingestion.validate(frame, "x")
    assert any(i.check == "extra_column" and i.severity == "info" for i in rep.issues)
    assert not any(i.check == "domain" for i in rep.issues)
    cleaned = ingestion.clean(frame, rep)
    assert "notes" not in cleaned.columns
    assert set(cleaned["school"]) <= {"GP", "MS"}


def test_duplicates_and_g3_zero_warnings(frame):
    frame.loc[0, "G3"] = 0
    frame = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
    rep = ingestion.validate(frame, "x")
    checks = {i.check: i for i in rep.issues}
    assert checks["duplicate_rows"].n_rows == 1
    assert checks["g3_zero"].n_rows == int((frame["G3"] == 0).sum())
    assert set(rep.to_frame().columns) == {"severity", "check", "column", "message", "n_rows"}


def test_missing_summary(frame):
    frame.loc[[0, 1], "age"] = np.nan
    ms = ingestion.missing_summary(frame)
    top = ms.iloc[0]
    assert top["column"] == "age"
    assert top["n_missing"] == 2
    assert top["pct_missing"] == pytest.approx(2 / len(frame) * 100)
    assert ingestion.missing_summary(frame.iloc[0:0])["pct_missing"].sum() == 0


# ----------------------------------------------------------- derived & merging
def test_absence_band_edges():
    s = pd.Series([0, 1, 5, 6, 10, 11, 20, 21, 93])
    bands = ingestion.absence_band(s).astype(str).tolist()
    assert bands == ["0", "1–5", "1–5", "6–10", "6–10", "11–20", "11–20", "20+", "20+"]


def test_pass_threshold_and_flags(frame):
    frame.loc[0, "G3"], frame.loc[1, "G3"], frame.loc[2, "G3"] = 9, 10, 0
    df, _ = ingestion.prepare({"mat": frame})
    assert df.loc[0, "pass"] == 0
    assert df.loc[1, "pass"] == 1
    assert bool(df.loc[2, config.G3_ZERO_FLAG_COL])
    assert not df[config.BOTH_SUBJECTS_FLAG_COL].any()  # single subject


def test_cross_subject_matching():
    mat = make_frame(20, seed=1)
    por = make_frame(30, seed=2)
    por.loc[:4, list(config.STUDENT_MATCH_KEYS)] = mat.loc[
        :4, list(config.STUDENT_MATCH_KEYS)
    ].values
    df, reports = ingestion.prepare({"mat": mat, "por": por})
    assert set(reports) == {"mat", "por"}
    assert list(df["subject"].unique()) == ["mat", "por"]
    flagged = df[df[config.BOTH_SUBJECTS_FLAG_COL]]
    assert (flagged["subject"] == "mat").sum() >= 5
    assert (flagged["subject"] == "por").sum() >= 5


def test_prepare_skips_rejected_tables(frame):
    df, reports = ingestion.prepare({"bad": frame.drop(columns=["G3"])})
    assert df.empty
    assert not reports["bad"].is_valid


def test_prepare_upload_with_and_without_subject(frame):
    df, _ = ingestion.prepare_upload(frame, "u.csv")
    assert set(df["subject"]) == {config.DEFAULT_UPLOAD_SUBJECT}
    frame["subject"] = ["mat"] * 20 + ["por"] * 20
    df, _ = ingestion.prepare_upload(frame, "u.csv")
    assert df["subject"].value_counts().to_dict() == {"mat": 20, "por": 20}
    bad, rep = ingestion.prepare_upload(frame.drop(columns=["sex"]), "u.csv")
    assert bad.empty
    assert not rep["u.csv"].is_valid


def test_load_bundled_and_roundtrip(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(FileNotFoundError, match="fetch_data"):
        ingestion.load_bundled(raw)
    make_frame(25, 3).to_csv(raw / "student-mat.csv", sep=";", index=False)
    make_frame(35, 4).to_csv(raw / "student-por.csv", sep=";", index=False)
    df, reports = ingestion.load_bundled(raw)
    assert len(df) == 60
    assert reports["mat"].n_rows == 25
    out = ingestion.save_processed(df, tmp_path / "proc" / "clean.csv")
    back = ingestion.load_processed(out)
    pd.testing.assert_frame_equal(back, df, check_dtype=False, check_categorical=False)
    assert back[config.ABSENCE_BAND_COL].cat.ordered


def test_quality_summary_values_are_computed(frame):
    frame.loc[[0, 1], "G3"] = 0
    df, reports = ingestion.prepare({"mat": frame, "por": make_frame(10, 9)})
    s = ingestion.data_quality_summary(df)
    assert s["total_rows"] == len(df)
    assert s["g3_zero_total"] == int((df["G3"] == 0).sum())
    assert s["pass_rate_all"] == pytest.approx(df["pass"].mean())
    nz = df[df["G3"] != 0]
    assert s["pass_rate_excl_g3_zero"] == pytest.approx(nz["pass"].mean())
    assert sum(s["absence_bands"].values()) == len(df)
    text = ingestion.format_quality_summary(s, reports)
    assert "DATA-QUALITY SUMMARY" in text
    assert "Mathematics" in text
