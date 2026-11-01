from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import analytics, config, ingestion
from tests.conftest import make_frame


@pytest.fixture
def data() -> pd.DataFrame:
    df, _ = ingestion.prepare({"mat": make_frame(60, 1), "por": make_frame(90, 2)})
    return df


def test_filters_default_is_noop(data):
    assert len(analytics.apply_filters(data, analytics.FilterSpec())) == len(data)


def test_filters_combine(data):
    spec = analytics.FilterSpec(
        subjects=("mat",), sexes=("F",), age_range=(16, 18), exclude_g3_zero=True
    )
    out = analytics.apply_filters(data, spec)
    expected = data[
        (data.subject == "mat") & (data.sex == "F") & data.age.between(16, 18) & (data.G3 != 0)
    ]
    assert out.index.equals(expected.index)


def test_filter_options(data):
    opts = analytics.filter_options(data)
    assert opts["subjects"] == ["mat", "por"]
    assert opts["age_range"] == (data.age.min(), data.age.max())


def test_kpis_match_pandas(data):
    k = analytics.kpis(data)
    assert k["n"] == len(data)
    assert k["mean_g3"] == pytest.approx(data.G3.mean())
    assert k["median_g3"] == pytest.approx(data.G3.median())
    assert k["pass_rate"] == pytest.approx((data.G3 >= 10).mean())
    assert k["pass_rate"] + k["fail_rate"] == pytest.approx(1)


def test_kpis_empty(data):
    k = analytics.kpis(data.iloc[0:0])
    assert k["n"] == 0
    assert math.isnan(k["pass_rate"])


def test_pass_rate_by_keeps_empty_categories(data):
    sub = data[data.absences > 20]
    rates = analytics.pass_rate_by(sub, config.ABSENCE_BAND_COL)
    assert list(rates[config.ABSENCE_BAND_COL]) == list(config.ABSENCE_BAND_LABELS)
    assert rates.loc[0, "n"] == 0
    assert np.isnan(rates.loc[0, "pass_rate"])
    last = rates.iloc[-1]
    assert last["n"] == len(sub)
    assert last["pass_rate"] == pytest.approx(sub["pass"].mean())


def test_mean_by(data):
    m = analytics.mean_by(data, "studytime")
    assert list(m["studytime"]) == sorted(data.studytime.unique())
    row = m.iloc[0]
    grp = data[data.studytime == row["studytime"]].G3
    assert row["n"] == len(grp)
    assert row["mean"] == pytest.approx(grp.mean())
    assert row["sem"] == pytest.approx(grp.std() / np.sqrt(len(grp)))


def test_subject_summary_and_descriptives(data):
    s = analytics.subject_summary(data).set_index("subject")
    assert s.loc["mat", "students"] == 60
    assert s.loc["por", "pass_rate"] == pytest.approx(data[data.subject == "por"]["pass"].mean())
    d = analytics.descriptive_stats(data).set_index("attribute")
    assert d.loc["G3", "mean"] == pytest.approx(data.G3.mean())
    assert d.loc["G3", "count"] == len(data)


def test_correlations(data):
    data = data.copy()
    data["G2"] = data["G3"]  # perfect correlation
    corr = analytics.correlations_with(data)
    assert "G3" not in corr.index
    assert corr["G2"] == pytest.approx(1.0)
    assert corr.is_monotonic_increasing
    m = analytics.correlation_matrix(data)
    assert np.diag(m) == pytest.approx(1.0)
    assert analytics.correlations_with(data.iloc[:2]).empty


def test_point_counts():
    df = pd.DataFrame({"G2": [1, 1, 2], "G3": [5, 5, 6]})
    assert list(analytics.point_counts(df, "G2", "G3")) == [2, 2, 1]
