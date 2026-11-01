"""Early Warning: students predicted to fail, with probability, risk tier and reasons."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, explain, modeling, storage, ui, visuals

ui.setup_page("Early Warning")
ui.page_header(
    "Early Warning",
    "Students the saved model predicts are at risk of failing (G3 < 10), with the probability, "
    "a risk tier and the three factors that raise each student's risk most.",
    eyebrow="Step 5 · Action",
)

df = ui.filtered_dataset()
full, _ = ui.get_active_dataset()
version = st.session_state.get(ui.DATASET_VERSION_KEY, 0)
filters_key = repr(st.session_state.get(ui.FILTER_KEY))

# ------------------------------------------------------------------ model
bundle = modeling.load_latest()
if bundle is not None:
    needed = modeling.feature_columns_from(bundle.pipeline)
    missing = [c for c in needed[0] + needed[1] if c not in df.columns]
    if missing:
        ui.friendly_error(
            "The saved model needs columns that this data does not have: " + ", ".join(missing)
        )
        bundle = None

if bundle is None:
    ui.empty_state(
        "No model saved yet. Save one on the Model Evaluation page, or train the recommended "
        "early-warning model (Random Forest, setting B, threshold 0.5) on the current data."
    )
    ok, reason = modeling.can_model(df)
    if not ok:
        st.caption(reason)
    elif st.button("Train recommended model", type="primary", icon=":material/model_training:"):
        with st.spinner("Cross-validating and fitting Random Forest (setting B)…"):
            cv = modeling.cross_validate(df, "B", "random_forest")
            pipe = modeling.fit(df, "B", "random_forest")
            modeling.save_model(
                pipe, "B", "random_forest", None, config.DEFAULT_THRESHOLD, cv.summary(), len(df)
            )
        st.rerun()
    st.stop()

meta = bundle.metadata
cv_recall = float(meta["metrics_cv"].get("fail_recall", 0))
st.markdown(
    ui.badge_html(
        f"Model: {meta['model_label']} · setting {meta['setting']} "
        f"({meta['setting_label']}) · CV fail recall {cv_recall:.1%}",
        "neutral",
    ),
    unsafe_allow_html=True,
)
st.caption(
    f"Trained on {meta['n_training_rows']:,} rows, saved {meta['saved_at']}. When it is applied "
    "to the same students it was trained on, its probabilities are optimistic; score new "
    "cohorts by uploading their data."
)

c1, c2 = st.columns([2, 1])
threshold = c1.slider(
    "Flag students when P(fail) is at least",
    0.05,
    0.95,
    float(meta["threshold"]),
    0.01,
    help="Starts at the threshold saved with the model.",
)
show_all = c2.toggle("Show all students, not only those flagged", value=False)


@st.cache_data(show_spinner=False, max_entries=16)
def _score(
    _pipe, _df: pd.DataFrame, _ref: pd.DataFrame, key: tuple, threshold: float, explain_all: bool
) -> pd.DataFrame:
    return explain.score(_pipe, _df, threshold, _ref, explain_all)


with st.spinner("Scoring students and computing reasons…"):
    try:
        scored = _score(
            bundle.pipeline, df, full, (meta["file"], version, filters_key), threshold, show_all
        )
    except Exception:  # never show a stack trace
        ui.friendly_error("Scoring failed for this data. Try a different model or filters.")
        st.stop()

counts = explain.tier_counts(scored["risk_tier"])
n_flagged = int(scored["predicted_fail"].sum())

run_key = (meta["file"], version, filters_key, round(threshold, 2))
logged = st.session_state.setdefault("logged_runs", set())
if run_key not in logged:
    try:
        storage.log_prediction_run(
            meta["file"],
            meta["model_label"],
            meta["setting"],
            threshold,
            len(scored),
            n_flagged,
            counts,
        )
        logged.add(run_key)
    except Exception:  # logging is best-effort
        st.toast("This prediction run could not be logged.", icon="⚠️")

ui.kpi_row(
    [
        ui.Kpi("Students scored", f"{len(scored):,}"),
        ui.Kpi("Flagged at risk", f"{n_flagged:,}", f"P(fail) ≥ {threshold:.2f}"),
        ui.Kpi(f"{config.RISK_ICONS['High']} High risk", f"{counts['High']:,}", "P(fail) ≥ 0.70"),
        ui.Kpi(f"{config.RISK_ICONS['Medium']} Medium risk", f"{counts['Medium']:,}", "0.40–0.70"),
        ui.Kpi(f"{config.RISK_ICONS['Low']} Low risk", f"{counts['Low']:,}", "P(fail) < 0.40"),
    ]
)

left, right = st.columns([2, 3], gap="medium")
with left:
    ui.chart_title("Students by risk tier", "Every tier has an icon, a label and a pattern.")
    ui.chart(visuals.risk_donut(counts), key="ew_donut")
with right:
    ui.chart_title("Distribution of predicted risk", "Dotted line = flag threshold.")
    ui.chart(visuals.risk_histogram(scored["p_fail"], threshold), key="ew_hist")

# ------------------------------------------------------------------ table
ui.section(
    "Students at risk" if not show_all else "All students",
    "Reasons are the factors that raise each student's predicted risk most compared with a "
    "typical student. They describe the model, not proven causes.",
)
f1, f2 = st.columns([2, 1])
query = f1.text_input("Search", placeholder="Record id, subject, reason… e.g. 'absences'")
tiers = f2.multiselect(
    "Risk tier",
    list(explain.TIERS),
    default=list(explain.TIERS),
    format_func=lambda t: f"{config.RISK_ICONS[t]} {t}",
)
view = explain.filter_scored(
    scored,
    tuple(tiers) if set(tiers) != set(explain.TIERS) else (),
    query,
    only_flagged=not show_all,
)
if tiers == []:
    view = view.iloc[0:0]

table = view.copy()
table["risk_tier"] = table["risk_tier"].astype(str).map(lambda t: f"{config.RISK_ICONS[t]} {t}")
table[config.SUBJECT_COL] = table[config.SUBJECT_COL].map(
    lambda s: visuals.value_label(config.SUBJECT_COL, s)
)
show_cols = [
    config.RECORD_ID_COL,
    config.SUBJECT_COL,
    "school",
    "sex",
    "age",
    "p_fail",
    "risk_tier",
    "reason_1",
    "reason_2",
    "reason_3",
]
if view.empty:
    ui.empty_state("No students match this search and tier selection.")
else:
    st.dataframe(
        table[show_cols],
        hide_index=True,
        width="stretch",
        height=min(620, 38 + 35 * len(table)),
        column_config={
            config.RECORD_ID_COL: "Record",
            config.SUBJECT_COL: "Subject",
            "school": "School",
            "sex": "Sex",
            "age": st.column_config.NumberColumn("Age", format="%d"),
            "p_fail": st.column_config.ProgressColumn(
                "P(fail)", format="%.2f", min_value=0.0, max_value=1.0
            ),
            "risk_tier": "Tier",
            "reason_1": "Reason 1",
            "reason_2": "Reason 2",
            "reason_3": "Reason 3",
        },
    )
    st.caption(f"Showing {len(view):,} student(s).")
    export = view.drop(columns=["top_features"], errors="ignore").copy()
    export["risk_tier"] = export["risk_tier"].astype(str)
    st.download_button(
        "Download as CSV",
        export.to_csv(index=False).encode("utf-8"),
        file_name="early_warning_students.csv",
        mime="text/csv",
        icon=":material/download:",
    )

# ------------------------------------------------------------------ history
ui.section("Prediction history", "Each distinct scoring run is logged with its tier counts.")
try:
    runs = storage.list_prediction_runs()
except Exception:  # never show a stack trace
    runs = pd.DataFrame()
if runs.empty:
    st.caption("No runs logged yet.")
else:
    st.dataframe(
        runs,
        hide_index=True,
        width="stretch",
        column_config={"threshold": st.column_config.NumberColumn("threshold", format="%.2f")},
    )
