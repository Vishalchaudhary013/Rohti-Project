"""Entry point / landing page for the Education Analytics Dashboard.

Run with: ``streamlit run app.py``. Full landing content is delivered in a later milestone.
"""

import streamlit as st

from src import config

st.set_page_config(page_title=config.APP_TITLE, page_icon=config.APP_ICON, layout="wide")
st.title(f"{config.APP_ICON} {config.APP_TITLE}")
st.info("Scaffolding in place (Milestone 1). Pages are added in later milestones.")
