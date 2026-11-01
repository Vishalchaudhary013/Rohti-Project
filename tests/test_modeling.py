from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, ingestion, modeling
from tests.conftest import make_frame


def _learnable(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic data where G2 and failures drive G3, so models can beat the baseline."""
    raw = make_frame(n, seed)
    rng = np.random.default_rng(seed)
    raw["G3"] = np.clip(raw["G2"] + rng.integers(-1, 2, n), 0, 20)
    raw.loc[raw["failures"] >= 3, "G3"] = np.minimum(raw.loc[raw["failures"] >= 3, "G3"], 8)
    df, _ = ingestion.prepare({"mat": raw.iloc[: n // 2], "por": raw.iloc[n // 2 :]})
    return df


@pytest.fixture(scope="module")
def data():
    return _learnable()


@pytest.fixture(scope="module")
def results(data):
    return modeling.evaluate_all(data)


def test_feature_settings(data):
    num_a, cat_a = modeling.feature_columns(data, "A")
    num_b, cat_b = modeling.feature_columns(data, "B")
    assert {"G1", "G2"} <= set(num_a)
    assert not {"G1", "G2"} & set(num_b)
    assert "G3" not in num_a + num_b
    assert "subject" in cat_a
    assert len(num_a) + len(cat_a) == 32 + 1  # 32 attributes + subject
    single = data[data.subject == "mat"]
    assert "subject" not in modeling.feature_columns(single, "A")[1]


def test_make_estimator_params():
    assert modeling.make_estimator("decision_tree").max_depth == config.DT_MAX_DEPTH
    rf = modeling.make_estimator("random_forest", "balanced")
    assert rf.n_estimators == config.RF_N_ESTIMATORS
    assert rf.class_weight == "balanced"
    with pytest.raises(ValueError):
        modeling.make_estimator("svm")


def test_can_model(data):
    assert modeling.can_model(data)[0]
    tiny = data[data["pass"] == 1].head(10)
    ok, msg = modeling.can_model(tiny)
    assert not ok and "at least" in msg
    assert not modeling.can_model(pd.DataFrame())[0]


def test_metrics_at_matches_definition():
    y = np.array([0, 0, 1, 1, 1])
    proba_pass = np.array([0.2, 0.6, 0.7, 0.9, 0.3])  # P(fail) = .8 .4 .3 .1 .7
    m = modeling.metrics_at(y, proba_pass, 0.5)  # predicted: 0 1 1 1 0
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["fail_recall"] == pytest.approx(1 / 2)
    assert m["fail_precision"] == pytest.approx(1 / 2)
    assert m["recall"] == pytest.approx(2 / 3)
    cm = modeling.confusion(y, proba_pass, 0.5)
    assert cm.tolist() == [[1, 1], [1, 2]]
    lower = modeling.metrics_at(y, proba_pass, 0.35)  # flags P(fail) >= .35: rows 0,1,4
    assert lower["fail_recall"] == 1.0


def test_cv_results_shape_and_baseline(results, data):
    assert len(results) == 8
    table = modeling.results_table(results)
    assert set(table.setting) == {"A", "B"}
    base = results[("A", modeling.BASELINE)].summary()
    majority = max(data["pass"].mean(), 1 - data["pass"].mean())
    assert base["accuracy"] == pytest.approx(majority, abs=0.02)
    assert base["roc_auc"] == pytest.approx(0.5)
    r = results[("A", "random_forest")]
    assert len(r.fold_metrics) == config.CV_FOLDS
    assert r.oof_proba.shape == (len(data),)
    assert ((r.oof_proba >= 0) & (r.oof_proba <= 1)).all()


def test_setting_a_beats_baseline_and_is_deterministic(results, data):
    for model in modeling.MODEL_NAMES:
        assert results[("A", model)].summary()["roc_auc"] > 0.8
    again = modeling.cross_validate(data, "A", "random_forest")
    np.testing.assert_allclose(again.oof_proba, results[("A", "random_forest")].oof_proba)


def test_markdown_and_curves(results):
    md = modeling.format_results_markdown(modeling.results_table(results))
    assert md.count("\n") == 9 and "Fail-class recall" in md
    r = results[("B", "logistic_regression")]
    curve = modeling.threshold_curve(r.y, r.oof_proba)
    assert curve["fail_recall"].is_monotonic_decreasing
    roc = modeling.roc_points(r.y, r.oof_proba)
    assert roc["fpr"].iloc[0] == 0 and roc["tpr"].iloc[-1] == 1


def test_interpretation(data):
    rf = modeling.fit(data, "A", "random_forest")
    imp = modeling.feature_importance(rf)
    assert imp.index[0] == "G2"
    assert imp.sum() == pytest.approx(1)
    lr = modeling.fit(data, "A", "logistic_regression")
    coef = modeling.standardized_coefficients(lr)
    assert coef["G2"] > 0
    assert modeling.predict_fail_proba(lr, data).shape == (len(data),)


def test_save_and_load(tmp_path, data):
    assert modeling.load_latest(tmp_path) is None
    pipe = modeling.fit(data, "B", "decision_tree", "balanced")
    path = modeling.save_model(
        pipe, "B", "decision_tree", "balanced", 0.4, {"f1": 0.5}, len(data), tmp_path
    )
    bundle = modeling.load_latest(tmp_path)
    assert bundle is not None and path.exists()
    meta = bundle.metadata
    assert meta["setting"] == "B" and meta["threshold"] == 0.4
    assert meta["params"]["max_depth"] == config.DT_MAX_DEPTH
    assert "G1" not in meta["numeric_features"]
    np.testing.assert_allclose(
        modeling.predict_fail_proba(bundle.pipeline, data),
        modeling.predict_fail_proba(pipe, data),
    )
    (tmp_path / "latest.json").write_text("{bad json")
    assert modeling.load_latest(tmp_path) is None


def test_single_class_training_fold_is_safe(data):
    only_pass = data[data["pass"] == 1]
    pipe = modeling.build_pipeline(only_pass, "B", modeling.BASELINE)
    pipe.fit(only_pass, modeling.target(only_pass))
    assert (modeling.predict_fail_proba(pipe, data) == 0).all()
