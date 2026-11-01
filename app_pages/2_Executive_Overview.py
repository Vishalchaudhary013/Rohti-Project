"""Executive Overview: headline KPIs, pass rates by subject and school, summary tables."""

from __future__ import annotations

import streamlit as st

from src import analytics, config, ui, visuals

ui.setup_page("Executive Overview")
ui.page_header(
    "Executive Overview",
    "Headline outcomes for the students matching the sidebar filters. "
    "A pass is a final grade (G3) of 10 or more out of 20.",
    eyebrow="Step 2 · Descriptive",
)

df = ui.filtered_dataset()
k = analytics.kpis(df)
ui.kpi_row(
    [
        ui.Kpi("Students (records)", f"{k['n']:,}", "One row per student per subject"),
        ui.Kpi("Mean final grade", f"{k['mean_g3']:.2f}", "G3, out of 20"),
        ui.Kpi("Median final grade", f"{k['median_g3']:.1f}", "G3, out of 20"),
        ui.Kpi("Pass rate", f"{k['pass_rate']:.1%}", "G3 ≥ 10"),
        ui.Kpi("Fail rate", f"{k['fail_rate']:.1%}", "G3 < 10"),
    ]
)

by_subject = analytics.pass_rate_by(df, config.SUBJECT_COL)
ui.kpi_row(
    [
        ui.Kpi(
            f"Pass rate · {visuals.value_label(config.SUBJECT_COL, row.subject)}",
            f"{row.pass_rate:.1%}" if row.n else "—",
            f"{int(row.n_pass):,} of {int(row.n):,} passed",
        )
        for row in by_subject.itertuples(index=False)
    ]
)

left, right = st.columns(2, gap="medium")
with left:
    ui.chart_title("Pass rate by subject", "Bar labels show the pass rate; n is the group size.")
    ui.chart(visuals.pass_rate_bar(by_subject, config.SUBJECT_COL), by_subject, key="ov_subject")
with right:
    by_school = analytics.pass_rate_by(df, "school")
    ui.chart_title("Pass rate by school")
    ui.chart(visuals.pass_rate_bar(by_school, "school"), by_school, key="ov_school")

ui.section("Subject-wise summary")
summary = analytics.subject_summary(df)
summary[config.SUBJECT_COL] = summary[config.SUBJECT_COL].map(
    lambda s: visuals.value_label(config.SUBJECT_COL, s)
)
st.dataframe(
    summary,
    hide_index=True,
    width="stretch",
    column_config={
        "subject": "Subject",
        "students": st.column_config.NumberColumn("Students", format="%d"),
        "mean_g3": st.column_config.NumberColumn("Mean G3", format="%.2f"),
        "median_g3": st.column_config.NumberColumn("Median G3", format="%.1f"),
        "pass_rate": st.column_config.ProgressColumn(
            "Pass rate", format="percent", min_value=0.0, max_value=1.0
        ),
        "g3_zero": st.column_config.NumberColumn("G3 = 0 rows", format="%d"),
    },
)

ui.section("Descriptive statistics", "Grades, age, absences, past failures and study time.")
st.dataframe(
    analytics.descriptive_stats(df),
    hide_index=True,
    width="stretch",
    column_config={
        c: st.column_config.NumberColumn(c, format="%.2f")
        for c in ("mean", "std", "min", "25%", "50%", "75%", "max")
    },
)
