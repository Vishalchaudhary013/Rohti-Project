"""Methodology: data source, pipeline, metric definitions and limitations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, modeling, ui

ui.setup_page("Methodology")
ui.page_header(
    "Methodology",
    "How the data is prepared, how models are evaluated, what every metric means and "
    "what this dashboard cannot tell you.",
    eyebrow="Reference",
)

# ------------------------------------------------------------------ dataset
ui.section("Dataset")
st.markdown(
    f"""
**UCI Student Performance** ([dataset page]({config.DATASET_PAGE_URL})). Two secondary schools
in Portugal (Gabriel Pereira and Mousinho da Silveira); one file for Mathematics
({config.EXPECTED_ROWS["mat"]} rows) and one for Portuguese ({config.EXPECTED_ROWS["por"]} rows),
semicolon-delimited, {len(config.SCHEMA)} attributes per student: demographics, family background,
study habits, social life and three period grades G1, G2, G3 on a 0–20 scale.

> {config.DATASET_CITATION}

You can also upload your own CSV/XLSX files with the same 33 columns
(download the template on the Data Ingestion page).
"""
)

# ------------------------------------------------------------------ pipeline
ui.section("Pipeline")
steps = [
    (
        "1 · Ingest",
        "Read CSV (`;` or `,`) or XLSX. Each upload is logged in SQLite with its "
        "time, file name, row count and SHA-256 checksum.",
    ),
    (
        "2 · Validate",
        "Check required columns, numeric types, whole numbers, value ranges "
        "(e.g. G1–G3 within 0–20) and allowed categories. Problems are reported, never hidden.",
    ),
    (
        "3 · Clean",
        "Drop and count rows that break a rule; reject a file that misses a "
        "required column. Nothing is imputed or silently changed.",
    ),
    (
        "4 · Derive",
        f"`pass` = 1 if G3 ≥ {config.PASS_THRESHOLD}; `absence_band` "
        f"({', '.join(config.ABSENCE_BAND_LABELS)}); flags for G3 = 0 and for students matched "
        "in both subject files.",
    ),
    (
        "5 · Describe",
        "KPIs, group pass rates and means (with group sizes and ±1 standard "
        "error) and Pearson correlations, all recomputed for the sidebar filters.",
    ),
    (
        "6 · Model",
        f"Logistic Regression, Decision Tree (max_depth={config.DT_MAX_DEPTH}) and "
        f"Random Forest (n_estimators={config.RF_N_ESTIMATORS}) after scaling numbers and one-hot "
        f"encoding categories; {config.CV_FOLDS}-fold StratifiedKFold (shuffle, "
        f"random_state={config.RANDOM_STATE}); benchmarked against the majority class.",
    ),
    (
        "7 · Explain",
        "Score students, assign a risk tier and derive the three factors that "
        "raise each student's predicted risk most. Each scoring run is logged in SQLite.",
    ),
]
cols = st.columns(len(steps) // 2 + 1)
for i, (title, body) in enumerate(steps):
    with cols[i % len(cols)], st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(body)

ui.section("Feature settings")
st.markdown(
    f"""
- **Setting A — {config.FEATURE_SETTINGS["A"]}.** All 32 non-target attributes, including the
  first- and second-period grades. Highly accurate, but only available late in the year.
- **Setting B — {config.FEATURE_SETTINGS["B"]}.** G1 and G2 removed. This is the realistic
  early-warning case: predict risk before any period grade exists.
- When both subjects are loaded, `subject` is added as an input. G3 and anything derived from
  it (`pass`, flags) are never inputs.
"""
)

# ------------------------------------------------------------------ metrics
ui.section("Metric definitions", "Positive class for precision/recall/F1 is 'pass' unless noted.")
definitions = pd.DataFrame(
    [
        ("Accuracy", "Share of all students classified correctly."),
        ("Precision (pass)", "Of students predicted to pass, the share who passed."),
        ("Recall (pass)", "Of students who passed, the share predicted to pass."),
        ("F1 (pass)", "Harmonic mean of pass precision and recall."),
        (
            "ROC-AUC",
            "Probability a random passing student gets a higher P(pass) than a random "
            "failing one. 0.5 = chance; threshold-independent.",
        ),
        (
            "Fail-class recall",
            "Of students who actually failed, the share the model flagged. "
            "The key early-warning metric: missed students get no support.",
        ),
        (
            "Fail-class precision",
            "Of students flagged, the share who actually failed. Low "
            "precision means support is spread over students who would have passed.",
        ),
        (
            "Majority-class baseline",
            "Always predict the most common outcome. Any useful model "
            "must beat it, especially on fail-class recall, where the baseline scores 0 whenever "
            "most students pass.",
        ),
        (
            "Mean ± std",
            f"Each metric is computed per fold at threshold "
            f"{config.DEFAULT_THRESHOLD} and averaged over {config.CV_FOLDS} folds. The threshold "
            "slider recomputes metrics from pooled out-of-fold predictions.",
        ),
    ],
    columns=["Metric", "Definition"],
)
st.table(definitions.set_index("Metric"))

ui.section("Risk tiers and reason codes")
st.markdown(
    f"""
- **Tiers** on predicted P(fail): ▲ High ≥ {config.RISK_HIGH}, ◆ Medium
  {config.RISK_MEDIUM}–{config.RISK_HIGH}, ● Low < {config.RISK_MEDIUM}. A student is *flagged*
  when P(fail) is at or above the chosen threshold.
- **Reason codes** are model-agnostic "what-if" effects: each input is replaced by a typical
  value (median for numbers, most common value for categories) and the drop in P(fail) is
  measured. The {config.TOP_N_REASONS} largest increases (above one percentage point) are
  written in plain language from the student's actual values, e.g. "2 past failures" or
  "absences in top 10%". Models compared: {", ".join(modeling.MODEL_NAMES.values())}.
"""
)

# ------------------------------------------------------------------ limitations
ui.section("Limitations")
st.markdown(
    """
- **Small, specific sample.** Two schools in one Portuguese region, collected in 2005–2006.
  Patterns may not transfer to other countries, systems or years without re-validation.
- **Self-reported attributes.** Study time, alcohol use, going out and family relationships come
  from questionnaires and are coarse ordinal scales that may be misreported.
- **Correlation is not causation.** Charts, importances, coefficients and reason codes describe
  associations the data and the model found, not effects of interventions.
- **Class imbalance.** Most students pass, so accuracy flatters; always read fail-class recall
  and precision against the baseline. `class_weight="balanced"` and the threshold change the
  trade-off.
- **G3 = 0 rows.** A final grade of exactly 0 most likely means the student did not sit the
  final exam rather than scoring zero (check how many have non-zero G1/G2 in your data).
  They are kept by default and can be excluded with the sidebar
  toggle; results differ between the two choices.
- **Duplicated students.** Some students appear in both subject files. There is no student ID,
  so matches use 13 shared attributes and are approximate. Records are student–subject pairs;
  cross-validation folds may place the same student in training and test sets.
- **Scoring the training data.** On the Early Warning page the model usually scores the
  students it was trained on, so probabilities are optimistic. Use it on new cohorts.
"""
)

ui.section("Privacy and fairness")
st.markdown(
    """
- Student records are personal data. Upload only what you are authorised to process, prefer
  pseudonymous IDs, and keep exports (CSV, figures) within approved systems. Uploaded data
  stays in the server session and in local files under `data/`; nothing is sent elsewhere.
- Attributes such as sex, age, address, family background and relationships can encode social
  disadvantage. A flag must start a supportive conversation, never a penalty, and should be
  reviewed by a teacher who knows the student.
- Before real use, check error rates across groups (e.g. by sex, school and address) with the
  sidebar filters, and monitor the model over time.
"""
)
