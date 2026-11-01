"""Entry point / landing page for the Education Analytics Dashboard.

Run with: ``streamlit run app.py``.
"""

from __future__ import annotations

import streamlit as st

from src import config, ui


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
        counts = dataset[config.SUBJECT_COL].value_counts().sort_index()
        source = "Bundled UCI dataset" if meta.get("source") == "bundled" else "Uploaded data"
        st.markdown(ui.badge_html(f"{source} loaded", "good"), unsafe_allow_html=True)
        ui.kpi_row(
            [ui.Kpi("Records", f"{len(dataset):,}", "Student–subject rows")]
            + [ui.Kpi(config.SUBJECT_LABELS.get(s, s), f"{n:,}", "rows") for s, n in counts.items()]
            + [ui.Kpi("Files", f"{len(meta.get('files', []))}", "Loaded this session")]
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
    st.navigation(NAV_PAGES).run()
