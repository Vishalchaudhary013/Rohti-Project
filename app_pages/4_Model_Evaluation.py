"""Model Evaluation: cross-validated comparison of settings A and B against the baseline."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, modeling, ui, visuals

ui.setup_page("Model Evaluation")
ui.page_header(
    "Model Evaluation",
    "Three models are cross-validated (5 stratified folds, seed 42) with prior grades "
    "(setting A) and without them (setting B, the early-warning case). Every model is "
    "benchmarked against always predicting the majority class.",
    eyebrow="Step 4 · Predictive",
)


@st.cache_data(show_spinner=False, max_entries=8)
def _evaluate(df: pd.DataFrame, class_weight: str | None) -> dict:
    return modeling.evaluate_all(df, class_weight)  # type: ignore[arg-type]


@st.cache_resource(show_spinner=False, max_entries=8)
def _fit(df: pd.DataFrame, setting: str, name: str, class_weight: str | None):
    return modeling.fit(df, setting, name, class_weight)  # type: ignore[arg-type]


df = ui.filtered_dataset()
ok, reason = modeling.can_model(df)
if not ok:
    ui.empty_state(reason)
    st.stop()

balanced = st.toggle(
    'Use class_weight="balanced"',
    help="Gives failing students more weight during training, usually raising fail-class "
    "recall at the cost of precision.",
)
class_weight = "balanced" if balanced else None

with st.spinner("Cross-validating 8 model runs… (cached after the first run)"):
    try:
        results = _evaluate(df, class_weight)
    except Exception:  # never show a stack trace
        ui.friendly_error("Model evaluation failed on this data. Try widening the filters.")
        st.stop()
table = modeling.results_table(results)

# ------------------------------------------------------------------ comparison
ui.section(
    "Comparison of models",
    f"Mean ± std across {config.CV_FOLDS} folds at threshold 0.5 on {len(df):,} rows. "
    "The first row of each table is the majority-class baseline.",
)


def _fmt(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"Model": frame["model"]})
    for metric, label in modeling.METRIC_LABELS.items():
        out[label] = [
            f"{m:.3f} ± {s:.3f}" if pd.notna(m) else "n/a"
            for m, s in zip(frame[metric], frame[f"{metric}_std"], strict=True)
        ]
    return out


styled = _fmt(table)
styled.insert(0, "Setting", [f"{s} · {config.FEATURE_SETTINGS[s]}" for s in table["setting"]])
st.dataframe(styled, hide_index=True, width="stretch")

metric = st.selectbox(
    "Compare models on",
    list(modeling.METRIC_LABELS),
    index=list(modeling.METRIC_LABELS).index("fail_recall"),
    format_func=modeling.METRIC_LABELS.get,
)
ui.chart(
    visuals.metric_comparison_bar(table, metric, modeling.METRIC_LABELS[metric]),
    table[["setting", "model", metric, f"{metric}_std"]],
    key="me_compare",
)

# ------------------------------------------------------------------ inspect
ui.section(
    "Inspect one model",
    "Move the threshold to trade fail-class recall (catching at-risk students) against "
    "precision (not flagging students who would pass). Computed from out-of-fold predictions.",
)
c1, c2, c3 = st.columns([1, 1, 2])
model_key = c1.selectbox(
    "Model", list(modeling.MODEL_NAMES), format_func=modeling.MODEL_NAMES.get, index=2
)
setting = c2.selectbox(
    "Setting",
    list(config.FEATURE_SETTINGS),
    index=1,
    format_func=lambda s: f"{s} · {config.FEATURE_SETTINGS[s]}",
)
threshold = c3.slider(
    "Decision threshold on P(fail)",
    0.05,
    0.95,
    config.DEFAULT_THRESHOLD,
    0.01,
    help="A student is flagged as at risk when the predicted probability of failing is at "
    "least this value.",
)

res = results[(setting, model_key)]
base = results[(setting, modeling.BASELINE)]
live = modeling.metrics_at(res.y, res.oof_proba, threshold)
base_live = modeling.metrics_at(base.y, base.oof_proba, threshold)
flagged = float((1 - res.oof_proba >= threshold).mean())
ui.kpi_row(
    [
        ui.Kpi("Accuracy", f"{live['accuracy']:.1%}", f"Baseline {base_live['accuracy']:.1%}"),
        ui.Kpi("Fail-class recall", f"{live['fail_recall']:.1%}", "Failing students caught"),
        ui.Kpi("Fail-class precision", f"{live['fail_precision']:.1%}", "Flags that were right"),
        ui.Kpi("ROC-AUC", f"{live['roc_auc']:.3f}", "Threshold-independent"),
        ui.Kpi("Students flagged", f"{flagged:.1%}", f"At threshold {threshold:.2f}"),
    ]
)

left, right = st.columns(2, gap="medium")
with left:
    ui.chart_title(f"Confusion matrix · {res.label}, setting {setting}")
    cm = modeling.confusion(res.y, res.oof_proba, threshold)
    ui.chart(visuals.confusion_heatmap(cm), key="me_cm")
with right:
    ui.chart_title(f"ROC curves · setting {setting}", "Detecting failing students.")
    curves, aucs = {}, {}
    for key, label in modeling.MODEL_NAMES.items():
        r = results[(setting, key)]
        curves[label] = modeling.roc_points(r.y, r.oof_proba)
        aucs[label] = modeling.metrics_at(r.y, r.oof_proba)["roc_auc"]
    ui.chart(visuals.roc_chart(curves, aucs), key="me_roc")

curve = modeling.threshold_curve(res.y, res.oof_proba)
ui.chart_title("Threshold trade-off", "Lower thresholds catch more failing students but flag more.")
ui.chart(visuals.threshold_tradeoff(curve, threshold), curve, key="me_tradeoff")

# ------------------------------------------------------------------ drivers
ui.section(
    f"What drives the predictions · setting {setting}",
    "Fitted on all filtered rows. Importance shows how much the forest relies on a feature; "
    "coefficients show direction (positive = more likely to pass) per standard deviation.",
)
left, right = st.columns(2, gap="medium")
with st.spinner("Fitting models on all rows…"):
    rf = _fit(df, setting, "random_forest", class_weight)
    lr = _fit(df, setting, "logistic_regression", class_weight)
with left:
    ui.chart_title("Feature importance · Random Forest", "Top 15.")
    imp = modeling.feature_importance(rf)
    ui.chart(
        visuals.importance_bar(imp, "Mean decrease in impurity"),
        imp.rename("importance").rename_axis("feature").reset_index(),
        key="me_imp",
    )
with right:
    ui.chart_title("Standardised coefficients · Logistic Regression", "Top 15 by size.")
    coef = modeling.standardized_coefficients(lr)
    ui.chart(
        visuals.importance_bar(coef, "Log-odds of passing", signed=True),
        coef.rename("coefficient").rename_axis("feature").reset_index(),
        key="me_coef",
    )

# ------------------------------------------------------------------ save
ui.section(
    "Use this model for Early Warning",
    "Saves the selected model (fitted on all filtered rows) with its setting, parameters, "
    "threshold and cross-validated metrics.",
)
if st.button(f"Save {res.label} · setting {setting}", type="primary", icon=":material/save:"):
    try:
        pipe = _fit(df, setting, model_key, class_weight)
        path = modeling.save_model(
            pipe,
            setting,
            model_key,
            class_weight,
            threshold,
            res.summary(),
            len(df),  # type: ignore[arg-type]
        )
        st.session_state["saved_model_file"] = path.name
        st.success(f"Saved as {path.name}. The Early Warning page will use it.", icon="✅")
    except OSError:
        ui.friendly_error("The model could not be saved to disk.")

latest = modeling.load_latest()
if latest is not None:
    m = latest.metadata
    st.caption(
        f"Current Early Warning model: {m['model_label']} · setting {m['setting']} · "
        f"threshold {m['threshold']:.2f} · class weight {m['class_weight'] or 'none'} · "
        f"saved {m['saved_at']}"
    )
