"""Entry point / landing page for the Education Analytics Dashboard.

Run with: ``streamlit run app.py``.
"""

from __future__ import annotations

import streamlit as st

from src import analytics, config, ui


def landing() -> None:
    """Render the landing page: summary, dataset status and navigation guide."""
    ui.page_header(
        config.APP_TITLE,
        "Descriptive, diagnostic and predictive analytics for secondary-school student "
        "performance, with an interpretable early-warning model that flags students at risk "
        "of failing and explains why.",
        eyebrow="Education analytics",
    )

    dataset, meta = ui.get_active_dataset()

    ui.section("Dataset status")
    if dataset is not None:
        source = "Bundled UCI dataset" if meta.get("source") == "bundled" else "Uploaded data"
        st.markdown(ui.badge_html(f"{source} loaded", "good"), unsafe_allow_html=True)
        k = analytics.kpis(dataset)
        subjects = ", ".join(
            config.SUBJECT_LABELS.get(s, s) for s in sorted(dataset[config.SUBJECT_COL].unique())
        )
        ui.kpi_row(
            [
                ui.Kpi("Students (records)", f"{k['n']:,}", subjects),
                ui.Kpi("Average final grade", f"{k['mean_g3']:.2f} / 20", "Mean of G3"),
                ui.Kpi("Overall pass rate", f"{k['pass_rate']:.1%}", "G3 ≥ 10"),
                ui.Kpi("Files", f"{len(meta.get('files', []))}", "Loaded this session"),
            ]
        )
    else:
        st.markdown(ui.badge_html("No dataset loaded", "warning"), unsafe_allow_html=True)
        ui.empty_state("Upload your student CSV or XLSX files to get started.")
        st.page_link(
            "app_pages/1_Data_Ingestion.py", label="Upload data", icon=":material/upload_file:"
        )

    ui.section("How to use this dashboard", "Work through the pages in order, left to right.")
    guide = [
        (
            "app_pages/1_Data_Ingestion.py",
            "Data Ingestion",
            "Use the bundled data or upload CSV/XLSX; review validation and history.",
        ),
        (
            "app_pages/2_Executive_Overview.py",
            "Executive Overview",
            "Headline KPIs: grades, pass and fail rates by subject.",
        ),
        (
            "app_pages/3_Exploratory_Analysis.py",
            "Exploratory Analysis",
            "Interactive charts of what relates to final grades.",
        ),
        (
            "app_pages/4_Model_Evaluation.py",
            "Model Evaluation",
            "Compare models with and without prior grades against a baseline.",
        ),
        (
            "app_pages/5_Early_Warning.py",
            "Early Warning",
            "Students at risk, with probability, tier and top reasons.",
        ),
        (
            "app_pages/6_Methodology.py",
            "Methodology",
            "Data source, pipeline, metric definitions and limitations.",
        ),
    ]
    cols = st.columns(3)
    for i, (path, title, body) in enumerate(guide):
        with cols[i % 3], st.container(border=True):
            st.markdown(
                ui.nav_card(f"{i + 1:02d}", title, body),
                unsafe_allow_html=True,
            )
            st.page_link(path, label=f"Open {title}", icon=":material/arrow_forward:")

    st.caption(f"Data: {config.DATASET_CITATION}")


NO_FILTER_PAGES = {"Home", "Data Ingestion", "Methodology"}

NAV_PAGES = {
    "Overview": [
        st.Page(landing, title="Home", icon=":material/home:", default=True),
        st.Page(
            "app_pages/1_Data_Ingestion.py", title="Data Ingestion", icon=":material/upload_file:"
        ),
    ],
    "Analytics": [
        st.Page(
            "app_pages/2_Executive_Overview.py",
            title="Executive Overview",
            icon=":material/dashboard:",
        ),
        st.Page(
            "app_pages/3_Exploratory_Analysis.py",
            title="Exploratory Analysis",
            icon=":material/query_stats:",
        ),
    ],
    "Prediction": [
        st.Page(
            "app_pages/4_Model_Evaluation.py", title="Model Evaluation", icon=":material/analytics:"
        ),
        st.Page("app_pages/5_Early_Warning.py", title="Early Warning", icon=":material/warning:"),
    ],
    "Reference": [
        st.Page("app_pages/6_Methodology.py", title="Methodology", icon=":material/menu_book:"),
    ],
}

if __name__ == "__main__":
    ui.setup_page("Home")
    page = st.navigation(NAV_PAGES)
    active, _ = ui.get_active_dataset()
    if active is not None and page.title not in NO_FILTER_PAGES:
        ui.render_filters(active)
    page.run()
