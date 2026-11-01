"""Exploratory Analysis: interactive charts of what relates to final grades."""

from __future__ import annotations

import contextlib
import io
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from src import analytics, config, ui, visuals

ui.setup_page("Exploratory Analysis")
ui.page_header(
    "Exploratory Analysis",
    "How final grades (G3) vary with study habits, past failures, absences and earlier "
    "grades. Hover any chart for exact values; open 'Show data' for the underlying table.",
    eyebrow="Step 3 · Diagnostic",
)

df = ui.filtered_dataset()
grades_tab, habits_tab, rel_tab, corr_tab = st.tabs(
    ["Grades", "Study habits", "Relationships", "Correlations"]
)

with grades_tab:
    ui.chart_title(
        "Distribution of final grades",
        "Fail bars (G3 < 10) are orange with a diagonal hatch; pass bars are solid blue.",
    )
    ui.chart(visuals.g3_histogram(df), key="ea_hist")
    left, right = st.columns(2, gap="medium")
    with left:
        ui.chart_title("Final grade by subject", "Box = middle 50%; dashed line = mean.")
        ui.chart(visuals.g3_box(df, config.SUBJECT_COL), key="ea_box_subject")
    with right:
        ui.chart_title("Final grade by higher-education aspiration")
        ui.chart(visuals.g3_box(df, "higher"), key="ea_box_higher")

with habits_tab:
    left, right = st.columns(2, gap="medium")
    with left:
        study = analytics.mean_by(df, "studytime")
        ui.chart_title(
            "Mean final grade by weekly study time", "Error bars show ±1 standard error."
        )
        ui.chart(visuals.mean_bar(study, "studytime"), study, key="ea_study")
    with right:
        fails = analytics.mean_by(df, "failures")
        ui.chart_title("Mean final grade by past failures", "Error bars show ±1 standard error.")
        ui.chart(visuals.mean_bar(fails, "failures"), fails, key="ea_fail")
    absence = analytics.pass_rate_by(df, config.ABSENCE_BAND_COL)
    ui.chart_title("Pass rate by number of absences")
    ui.chart(visuals.pass_rate_bar(absence, config.ABSENCE_BAND_COL), absence, key="ea_abs")

with rel_tab:
    ui.chart_title(
        "Second-period grade vs final grade",
        "Points are spread slightly (±0.25) so students with identical grades stay visible; "
        "hover shows exact grades and how many students share them. Fail = ✕, pass = ●.",
    )
    ui.chart(visuals.g2_g3_scatter(df), key="ea_scatter")

with corr_tab:
    corr = analytics.correlations_with(df)
    if corr.empty:
        ui.empty_state("Not enough variation in the filtered data to compute correlations.")
    else:
        left, right = st.columns([2, 3], gap="medium")
        with left:
            ui.chart_title("Correlation with the final grade", "Blue = positive, red = negative.")
            ui.chart(
                visuals.correlation_bar(corr),
                corr.rename("r").rename_axis("attribute").reset_index(),
                key="ea_corr_bar",
            )
        with right:
            matrix = analytics.correlation_matrix(df)
            ui.chart_title("Correlation heatmap", "Pearson r between numeric attributes.")
            ui.chart(visuals.correlation_heatmap(matrix), key="ea_heatmap")
        st.info(
            "Correlation shows association, not cause. Use these charts to ask better "
            "questions, then confirm them with domain knowledge or the models.",
            icon=":material/info:",
        )

ui.section(
    "Export figures",
    f"Renders all {len(visuals.FIGURE_CATALOG)} charts for the current filters as "
    f"{config.FIGURE_EXPORT_DPI}-dpi PNGs for reports.",
)
if st.button("Export all figures", icon=":material/image:"):
    with st.spinner("Rendering figures…"):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                paths = visuals.export_static_figures(df, Path(tmp))
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for p in paths:
                        zf.write(p, p.name)
            # Also keep a copy in reports/figures when running locally.
            with contextlib.suppress(OSError):
                visuals.export_static_figures(df, config.FIGURES_DIR)
            st.session_state["figures_zip"] = buf.getvalue()
        except Exception:  # never show a stack trace to the user
            ui.friendly_error("Figures could not be exported. Please try again.")
if st.session_state.get("figures_zip"):
    st.download_button(
        "Download figures (.zip)",
        st.session_state["figures_zip"],
        file_name="education_analytics_figures.zip",
        mime="application/zip",
        icon=":material/download:",
    )
    st.caption("Also saved to reports/figures/ on the server: " + ", ".join(visuals.FIGURE_CATALOG))
