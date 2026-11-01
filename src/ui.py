"""Presentation helpers shared by every page: global CSS, headers, KPI tiles, badges.

The visual language is flat: hairline borders instead of shadows, a 2px radius
("rounded-sm"), and colours taken only from the design tokens in :mod:`src.config`.
These helpers hold no business logic; they only render values passed in.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import streamlit as st

from src import config

StatusKind = Literal["good", "warning", "serious", "critical", "neutral"]

_STATUS_ICONS: dict[str, str] = {
    "good": "✓",
    "warning": "!",
    "serious": "!",
    "critical": "✕",
    "neutral": "•",
}


def _css() -> str:
    """Return the global stylesheet built from the design tokens."""
    r = f"{config.RADIUS_PX}px"
    return f"""
<style>
:root {{
  --ea-surface: {config.SURFACE};
  --ea-page: {config.PAGE_BG};
  --ea-ink: {config.INK_PRIMARY};
  --ea-ink-2: {config.INK_SECONDARY};
  --ea-muted: {config.INK_MUTED};
  --ea-border: {config.BORDER};
  --ea-accent: {config.ACCENT};
  --ea-radius: {r};
}}
/* Flat UI: no shadows anywhere, small radius. */
*, *::before, *::after {{ box-shadow: none !important; text-shadow: none !important; }}
button, input, textarea, select,
[data-baseweb="select"] > div, [data-baseweb="input"], [data-testid="stExpander"] details,
[data-testid="stFileUploaderDropzone"], [data-testid="stDataFrame"],
[data-testid="stMetric"], [data-testid="stAlert"], [data-baseweb="tag"],
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: var(--ea-radius) !important; }}
.block-container {{ padding-top: 2.25rem; max-width: 1280px; }}
h1, h2, h3 {{ color: var(--ea-ink); letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--ea-border); }}

.ea-header {{ border-bottom: 1px solid var(--ea-border); padding-bottom: 1rem;
  margin-bottom: 1.5rem; }}
.ea-eyebrow {{ color: var(--ea-accent); font-size: 0.75rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase; margin: 0 0 0.25rem; }}
.ea-header h1.ea-title {{ font-size: 1.75rem !important; font-weight: 650 !important;
  color: var(--ea-ink); margin: 0; padding: 0 !important; line-height: 1.25; }}
.ea-subtitle {{ color: var(--ea-ink-2); font-size: 0.95rem; margin: 0.4rem 0 0;
  max-width: 72ch; }}

.ea-grid {{ display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-bottom: 1rem; }}
.ea-card {{ background: var(--ea-surface); border: 1px solid var(--ea-border);
  border-radius: var(--ea-radius); padding: 14px 16px; }}
.ea-card p.ea-kpi-label {{ color: var(--ea-ink-2); font-size: 0.8rem !important; font-weight: 500;
  margin: 0; }}
.ea-card p.ea-kpi-value {{ color: var(--ea-ink); font-size: 1.75rem !important;
  font-weight: 650; margin: 0.2rem 0 0; line-height: 1.2; }}
.ea-card p.ea-kpi-caption {{ color: var(--ea-muted); font-size: 0.78rem !important;
  margin: 0.3rem 0 0; }}

.ea-section {{ margin: 1.75rem 0 0.75rem; }}
.ea-section h3 {{ font-size: 1.05rem; font-weight: 600; margin: 0; }}
.ea-section p {{ color: var(--ea-ink-2); font-size: 0.85rem; margin: 0.2rem 0 0; }}

.ea-badge {{ display: inline-flex; align-items: center; gap: 6px; font-size: 0.78rem;
  font-weight: 600; padding: 2px 8px; border-radius: var(--ea-radius);
  border: 1px solid var(--ea-border); background: var(--ea-surface); color: var(--ea-ink); }}
.ea-dot {{ width: 8px; height: 8px; border-radius: 1px; display: inline-block; }}

.ea-plain {{ padding: 2px 0 4px; }}
.ea-nav-title {{ font-weight: 600; color: var(--ea-ink); margin: 0 0 0.25rem; }}
.ea-nav-body {{ color: var(--ea-ink-2); font-size: 0.85rem; margin: 0; min-height: 3.2em; }}
.ea-step {{ color: var(--ea-muted); font-size: 0.75rem; font-weight: 600; margin: 0 0 0.4rem;
  font-variant-numeric: tabular-nums; }}
.ea-chart-title {{ font-weight: 600; color: var(--ea-ink); font-size: 0.95rem;
  margin: 0.5rem 0 0 !important; }}
p.ea-chart-cap {{ color: var(--ea-muted); font-size: 0.8rem !important; margin: 0.1rem 0 0; }}
[data-testid="stPlotlyChart"] {{ border: 1px solid var(--ea-border);
  border-radius: var(--ea-radius); background: var(--ea-surface); padding: 4px; }}
.ea-empty code {{ color: var(--ea-ink) !important; font-size: 0.85em; background: var(--ea-page);
  border: 1px solid var(--ea-border); border-radius: var(--ea-radius); padding: 1px 4px; }}
.ea-empty {{ border: 1px dashed var(--ea-border); border-radius: var(--ea-radius);
  padding: 28px; text-align: center; color: var(--ea-ink-2); background: var(--ea-surface); }}
</style>
"""


def setup_page(title: str) -> None:
    """Configure the page and inject the global stylesheet. Call first on every page."""
    st.set_page_config(
        page_title=f"{title} · {config.APP_TITLE}",
        page_icon=config.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_css(), unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    """Render a page title block with optional eyebrow and subtitle."""
    parts = ['<div class="ea-header">']
    if eyebrow:
        parts.append(f'<p class="ea-eyebrow">{html.escape(eyebrow)}</p>')
    parts.append(f'<h1 class="ea-title">{html.escape(title)}</h1>')
    if subtitle:
        parts.append(f'<p class="ea-subtitle">{html.escape(subtitle)}</p>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    """Render a section heading with an optional one-line caption."""
    cap = f"<p>{html.escape(caption)}</p>" if caption else ""
    st.markdown(
        f'<div class="ea-section"><h3>{html.escape(title)}</h3>{cap}</div>',
        unsafe_allow_html=True,
    )


@dataclass(frozen=True)
class Kpi:
    """A single stat tile: label, formatted value and optional caption."""

    label: str
    value: str
    caption: str = ""


def kpi_row(kpis: Sequence[Kpi]) -> None:
    """Render KPI tiles in a responsive, wrapping grid."""
    tiles = []
    for k in kpis:
        cap = f'<p class="ea-kpi-caption">{html.escape(k.caption)}</p>' if k.caption else ""
        tiles.append(
            '<div class="ea-card">'
            f'<p class="ea-kpi-label">{html.escape(k.label)}</p>'
            f'<p class="ea-kpi-value">{html.escape(k.value)}</p>{cap}</div>'
        )
    st.markdown(f'<div class="ea-grid">{"".join(tiles)}</div>', unsafe_allow_html=True)


def badge_html(label: str, kind: StatusKind = "neutral") -> str:
    """Return an icon + label status badge; colour never carries meaning alone."""
    color = config.STATUS_COLORS.get(kind, config.INK_MUTED)
    icon = _STATUS_ICONS[kind]
    return (
        f'<span class="ea-badge"><span class="ea-dot" style="background:{color}"></span>'
        f"{icon} {html.escape(label)}</span>"
    )


def badge(label: str, kind: StatusKind = "neutral") -> None:
    """Render a status badge."""
    st.markdown(badge_html(label, kind), unsafe_allow_html=True)


def nav_card(step: str, title: str, body: str) -> str:
    """Return the inner HTML for one navigation-guide card (placed in a bordered container)."""
    return (
        f'<div class="ea-plain"><p class="ea-step">{html.escape(step)}</p>'
        f'<p class="ea-nav-title">{html.escape(title)}</p>'
        f'<p class="ea-nav-body">{html.escape(body)}</p></div>'
    )


def empty_state(message: str) -> None:
    """Render a dashed placeholder panel; text wrapped in backticks renders as code."""
    parts = html.escape(message).split("`")
    body = "".join(f"<code>{s}</code>" if i % 2 else s for i, s in enumerate(parts))
    st.markdown(f'<div class="ea-empty">{body}</div>', unsafe_allow_html=True)


def friendly_error(message: str, detail: str = "") -> None:
    """Show a user-facing error without a stack trace."""
    st.error(message + (f"\n\n{detail}" if detail else ""), icon="⚠️")


DATASET_KEY = "dataset"
DATASET_META_KEY = "dataset_meta"
DATASET_VERSION_KEY = "dataset_version"
FILTER_KEY = "filters"


def set_active_dataset(dataset: object, meta: dict[str, object]) -> None:
    """Store the dataset every page reads, plus a description of where it came from."""
    st.session_state[DATASET_KEY] = dataset
    st.session_state[DATASET_META_KEY] = meta
    st.session_state[DATASET_VERSION_KEY] = st.session_state.get(DATASET_VERSION_KEY, 0) + 1


def get_active_dataset() -> tuple[object | None, dict[str, object]]:
    """Return ``(dataset, meta)`` for this session, or ``(None, {})`` if nothing is loaded."""
    return st.session_state.get(DATASET_KEY), st.session_state.get(DATASET_META_KEY, {})


def require_dataset() -> object:
    """Return the active dataset, or show an upload prompt and stop the page."""
    dataset, _ = get_active_dataset()
    if dataset is None:
        empty_state("No data loaded yet. Upload your student files on the Data Ingestion page.")
        st.page_link(
            "app_pages/1_Data_Ingestion.py",
            label="Go to Data Ingestion",
            icon=":material/upload_file:",
        )
        st.stop()
    return dataset


def bundled_available() -> bool:
    """Return True when both bundled UCI files are present in ``data/raw``."""
    return config.RAW_MAT_FILE.exists() and config.RAW_POR_FILE.exists()


def render_filters(df: pd.DataFrame) -> None:
    """Render the global sidebar filters and store the selection in session state.

    Widget keys include the dataset version so loading new data resets the filters.
    Call from the app entry point so selections persist across pages.
    """
    from src import analytics

    v = st.session_state.get(DATASET_VERSION_KEY, 0)
    opts = analytics.filter_options(df)
    sb = st.sidebar
    sb.markdown('<p class="ea-eyebrow">Filters</p>', unsafe_allow_html=True)

    def multi(label: str, name: str, column: str) -> tuple[str, ...]:
        choices = opts[name]
        picked = sb.multiselect(
            label,
            choices,
            default=choices,
            format_func=lambda c: _value_label(column, c),
            key=f"f_{name}_{v}",
        )
        return () if set(picked) == set(choices) else tuple(picked) or ("__none__",)

    subjects = multi("Subject", "subjects", config.SUBJECT_COL)
    schools = multi("School", "schools", "school")
    sexes = multi("Sex", "sexes", "sex")
    addresses = multi("Address", "addresses", "address")
    lo, hi = opts["age_range"]
    age = sb.slider("Age range", lo, hi, (lo, hi), key=f"f_age_{v}") if hi > lo else (lo, hi)
    exclude = sb.toggle("Exclude G3 = 0 (likely absent from final exam)", key=f"f_g3zero_{v}")
    st.session_state[FILTER_KEY] = analytics.FilterSpec(
        subjects=subjects,
        schools=schools,
        sexes=sexes,
        addresses=addresses,
        age_range=None if tuple(age) == (lo, hi) else (int(age[0]), int(age[1])),
        exclude_g3_zero=exclude,
    )


def _value_label(column: str, value: object) -> str:
    """Display label for a coded value (thin wrapper to avoid importing visuals here)."""
    if column == config.SUBJECT_COL:
        return config.SUBJECT_LABELS.get(str(value), str(value))
    return config.VALUE_LABELS.get(column, {}).get(value, str(value))


def filtered_dataset() -> pd.DataFrame:
    """Return the active dataset with the global filters applied (stops if none loaded)."""
    from src import analytics

    df = require_dataset()
    spec = st.session_state.get(FILTER_KEY) or analytics.FilterSpec()
    out = analytics.apply_filters(df, spec)
    active = spec != analytics.FilterSpec()
    if active:
        st.caption(f"Filters active: showing {len(out):,} of {len(df):,} rows.")
    if out.empty:
        empty_state("No rows match the current filters. Widen them in the sidebar.")
        st.stop()
    return out


def chart(fig: object, data: pd.DataFrame | None = None, key: str | None = None) -> None:
    """Render a Plotly figure full-width, with an optional 'Show data' table view."""
    st.plotly_chart(
        fig,
        width="stretch",
        key=key,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    if data is not None:
        with st.expander("Show data"):
            st.dataframe(data, hide_index=True, width="stretch")


def chart_title(title: str, caption: str = "") -> None:
    """Small heading placed directly above a chart."""
    cap = f'<p class="ea-chart-cap">{html.escape(caption)}</p>' if caption else ""
    st.markdown(f'<p class="ea-chart-title">{html.escape(title)}</p>{cap}', unsafe_allow_html=True)
