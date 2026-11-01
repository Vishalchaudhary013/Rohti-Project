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

    dataset, info = ui.load_bundled_dataset()

    ui.section("Dataset status")
    if info["status"] == "ready":
        counts = dataset[config.SUBJECT_COL].value_counts()
        st.markdown(ui.badge_html("Bundled dataset loaded", "good"), unsafe_allow_html=True)
        ui.kpi_row(
            [ui.Kpi("Records", f"{len(dataset):,}", "Student–subject rows")]
            + [
                ui.Kpi(config.SUBJECT_LABELS.get(s, s), f"{n:,}", "rows")
                for s, n in counts.sort_index().items()
            ]
            + [
                ui.Kpi("Attributes", f"{len(config.SCHEMA)}", "From the UCI codebook"),
            ]
        )
    else:
        st.markdown(ui.badge_html("No dataset loaded", "warning"), unsafe_allow_html=True)
        ui.empty_state(str(info.get("message", "")))

    ui.section("How to use this dashboard", "Work through the pages in order, left to right.")
    guide = [
        (
            "pages/1_Data_Ingestion.py",
            "Data Ingestion",
            "Use the bundled data or upload CSV/XLSX; review validation and history.",
        ),
        (
            "pages/2_Executive_Overview.py",
            "Executive Overview",
            "Headline KPIs: grades, pass and fail rates by subject.",
        ),
        (
            "pages/3_Exploratory_Analysis.py",
            "Exploratory Analysis",
            "Interactive charts of what relates to final grades.",
        ),
        (
            "pages/4_Model_Evaluation.py",
            "Model Evaluation",
            "Compare models with and without prior grades against a baseline.",
        ),
        (
            "pages/5_Early_Warning.py",
            "Early Warning",
            "Students at risk, with probability, tier and top reasons.",
        ),
        (
            "pages/6_Methodology.py",
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
        st.Page("pages/1_Data_Ingestion.py", title="Data Ingestion", icon=":material/upload_file:"),
    ],
    "Analytics": [
        st.Page(
            "pages/2_Executive_Overview.py", title="Executive Overview", icon=":material/dashboard:"
        ),
        st.Page(
            "pages/3_Exploratory_Analysis.py",
            title="Exploratory Analysis",
            icon=":material/query_stats:",
        ),
    ],
    "Prediction": [
        st.Page(
            "pages/4_Model_Evaluation.py", title="Model Evaluation", icon=":material/analytics:"
        ),
        st.Page("pages/5_Early_Warning.py", title="Early Warning", icon=":material/warning:"),
    ],
    "Reference": [
        st.Page("pages/6_Methodology.py", title="Methodology", icon=":material/menu_book:"),
    ],
}

if __name__ == "__main__":
    ui.setup_page("Home")
    st.navigation(NAV_PAGES).run()
