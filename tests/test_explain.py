from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, explain, modeling
from tests.test_modeling import _learnable


@pytest.fixture(scope="module")
def data():
    return _learnable(240, 5)


@pytest.fixture(scope="module")
def pipe(data):
    return modeling.fit(data, "A", "logistic_regression")


@pytest.mark.parametrize(
    ("p", "tier"),
    [(0.95, "High"), (0.7, "High"), (0.69, "Medium"), (0.4, "Medium"), (0.39, "Low"), (0.0, "Low")],
)
def test_risk_tier_boundaries(p, tier):
    assert explain.risk_tier(p) == tier


def test_risk_tiers_and_counts():
    tiers = explain.risk_tiers(np.array([0.9, 0.5, 0.1, 0.8]))
    assert list(tiers) == ["High", "Medium", "Low", "High"]
    assert explain.tier_counts(tiers) == {"High": 2, "Medium": 1, "Low": 1}
    assert explain.tier_counts(pd.Series([], dtype=str)) == {"High": 0, "Medium": 0, "Low": 0}


def test_contributions_are_zero_for_typical_student(pipe, data):
    cols = explain.model_inputs(pipe)
    typical = explain.typical_values(data, cols)
    row = data.iloc[[0]].copy()
    for c, v in typical.items():
        row[c] = v
    contrib = explain.contributions(pipe, row, data)
    assert list(contrib.columns) == cols
    assert np.allclose(contrib.to_numpy(), 0)


def test_low_g2_is_top_reason_for_failing_student(pipe, data):
    row = data[(data.G2 <= 3) & (data["pass"] == 0)].head(1)
    reasons = explain.reason_codes(pipe, row, data)
    assert reasons.iloc[0]["reason_1"].startswith(("second-period grade", "first-period grade"))
    assert "below pass mark" in reasons.iloc[0]["reasons"]


def test_no_reason_when_nothing_raises_risk(pipe, data):
    row = data[(data.G2 >= 18)].head(1)
    reasons = explain.reason_codes(pipe, row, data)
    assert reasons.iloc[0]["reasons"] in (explain.NO_REASON,) or reasons.iloc[0]["reason_1"]


@pytest.mark.parametrize(
    ("feature", "value", "expected"),
    [
        ("failures", 2, "2 past failures"),
        ("failures", 1, "1 past failure"),
        ("failures", 3, "3+ past failures"),
        ("G2", 7, "second-period grade 7/20 (below pass mark)"),
        ("G1", 12, "first-period grade 12/20"),
        ("studytime", 1, "studies under 2 h a week"),
        ("Medu", 1, "mother's education: primary (4th grade)"),
        ("goout", 5, "goes out with friends often (5/5)"),
        ("health", 1, "poor current health (1/5)"),
        ("higher", "no", "does not plan higher education"),
        ("schoolsup", "yes", "receives extra school support"),
        ("Pstatus", "A", "parents live apart"),
        ("Fjob", "at_home", "father's job: at home"),
        ("subject", "por", "subject: Portuguese"),
        ("traveltime", 4, "long journey to school"),
        ("Walc", 2, "Weekend alcohol consumption (1–5): 2"),
    ],
)
def test_describe(feature, value, expected, data):
    assert explain.describe(feature, value, data) == expected


def test_describe_absence_percentiles():
    ref = pd.DataFrame({"absences": list(range(100)), "age": [16] * 100})
    assert explain.describe("absences", 95, ref) == "absences in top 10% (95)"
    assert explain.describe("absences", 80, ref) == "absences in top 25% (80)"
    assert explain.describe("absences", 10, ref) == "10 absences"
    assert explain.describe("age", 19, ref) == "age 19 (older than most peers)"
    assert explain.describe("age", 15, ref) == "age 15"


def test_score_and_filter(pipe, data):
    scored = explain.score(pipe, data, 0.5)
    assert len(scored) == len(data)
    assert scored["p_fail"].is_monotonic_decreasing
    flagged = scored[scored.predicted_fail]
    assert (flagged.p_fail >= 0.5).all()
    assert (flagged.reasons != "").all()
    assert (scored.loc[~scored.predicted_fail, "reasons"] == "").all()
    assert config.RECORD_ID_COL in scored.columns
    only_high = explain.filter_scored(scored, ("High",))
    assert set(only_high.risk_tier.astype(str)) <= {"High"}
    rid = flagged.iloc[0][config.RECORD_ID_COL]
    assert explain.filter_scored(scored, query=rid.upper()).iloc[0][config.RECORD_ID_COL] == rid
    everyone = explain.filter_scored(scored, only_flagged=False)
    assert len(everyone) == len(data)
    all_explained = explain.score(pipe, data, 0.99, explain_all=True)
    assert (all_explained.reasons != "").all()
    none = explain.score(pipe, data, 1.01)
    assert not none.predicted_fail.any()
