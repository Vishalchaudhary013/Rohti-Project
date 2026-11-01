from __future__ import annotations

import pytest

from src import analytics, config, ingestion, visuals
from tests.conftest import make_frame


@pytest.fixture(scope="module")
def data():
    df, _ = ingestion.prepare({"mat": make_frame(60, 1), "por": make_frame(90, 2)})
    return df


def test_histogram_splits_pass_fail_with_pattern(data):
    fig = visuals.g3_histogram(data)
    names = [t.name for t in fig.data]
    assert names == [config.FAIL_LABEL, config.PASS_LABEL]
    fail, passed = fig.data
    assert sum(fail.y) + sum(passed.y) == len(data)
    assert fail.marker.pattern.shape == "/"
    assert max(fail.x) < config.PASS_THRESHOLD <= min(passed.x)


def test_rate_and_mean_bars_show_group_sizes(data):
    rates = analytics.pass_rate_by(data, config.SUBJECT_COL)
    fig = visuals.pass_rate_bar(rates, config.SUBJECT_COL)
    assert "n=60" in fig.layout.xaxis.ticktext[0]
    assert fig.data[0].text[0] == f"{rates.pass_rate[0]:.1%}"
    means = analytics.mean_by(data, "failures")
    fig = visuals.mean_bar(means, "failures")
    assert len(fig.data[0].y) == len(means)


def test_box_scatter_corr_heatmap(data):
    assert len(visuals.g3_box(data, "higher").data) == data.higher.nunique()
    sc = visuals.g2_g3_scatter(data)
    assert sum(len(t.x) for t in sc.data) == len(data)
    assert {t.marker.symbol for t in sc.data} == {"circle", "x-thin-open"}
    corr = analytics.correlations_with(data)
    assert len(visuals.correlation_bar(corr).data[0].x) == len(corr)
    hm = visuals.correlation_heatmap(analytics.correlation_matrix(data))
    assert hm.data[0].zmid == 0


def test_value_labels():
    assert visuals.value_label("subject", "mat") == "Mathematics"
    assert visuals.value_label("studytime", 1) == "<2 h"
    assert visuals.value_label("failures", 3) == "3+"
    assert visuals.value_label("unknown", 7) == "7"


def test_static_export_writes_all_figures(data, tmp_path):
    paths = visuals.export_static_figures(data, tmp_path, dpi=40)
    assert [p.name for p in paths] == list(visuals.FIGURE_CATALOG)
    assert all(p.stat().st_size > 1000 for p in paths)


def test_donuts(data):
    fig = visuals.outcome_donut(data)
    pie = fig.data[0]
    assert list(pie.labels) == [config.PASS_LABEL, config.FAIL_LABEL]
    assert sum(pie.values) == len(data)
    assert pie.marker.pattern.shape[1] == "/"
    assert f"{len(data):,}" in fig.layout.annotations[0].text
    sub = visuals.subject_donut(data)
    assert list(sub.data[0].labels) == ["Mathematics", "Portuguese"]


def test_model_figures():
    import numpy as np
    import pandas as pd

    from src import modeling

    table = pd.DataFrame(
        {
            "setting": ["A", "A", "B", "B"],
            "model": ["LR", "RF"] * 2,
            "f1": [0.8, 0.9, 0.6, 0.7],
            "f1_std": [0.01] * 4,
        }
    )
    assert len(visuals.metric_comparison_bar(table, "f1", "F1").data) == 2
    cm = np.array([[5, 3], [2, 10]])
    hm = visuals.confusion_heatmap(cm)
    assert hm.data[0].text[0][0].startswith("<b>5</b>")
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.4, 0.8, 0.6, 0.9, 0.3])
    roc = visuals.roc_chart({"LR": modeling.roc_points(y, p)}, {"LR": 0.9})
    assert "AUC 0.90" in roc.data[0].name
    trade = visuals.threshold_tradeoff(modeling.threshold_curve(y, p), 0.4)
    assert len(trade.data) == 2
    s = pd.Series([0.5, -0.3, 0.1], index=["a", "b", "c"])
    signed = visuals.importance_bar(s, "coef", signed=True)
    assert signed.data[0].text[-1] == "+0.50"
    assert visuals.importance_bar(s.abs(), "imp").data[0].marker.color == config.PASS_COLOR
