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


@st.cache_data(show_spinner="Loading dataset…")
def load_bundled_dataset() -> tuple[object, dict[str, object]]:
    """Load the bundled dataset once per session (processed CSV first, else raw files).

    Returns:
        ``(dataset, info)`` where ``dataset`` is a DataFrame (empty if unavailable) and
        ``info`` holds ``status`` (``"ready"``/``"missing"``/``"invalid"``) and ``message``.
    """
    import pandas as pd

    from src import ingestion

    if config.PROCESSED_FILE.exists():
        try:
            return ingestion.load_processed(), {"status": "ready", "source": "processed"}
        except (ValueError, KeyError, OSError):
            pass  # fall back to rebuilding from raw files
    try:
        dataset, reports = ingestion.load_bundled()
    except FileNotFoundError:
        return pd.DataFrame(), {
            "status": "missing",
            "message": "Bundled data not found. Run `python scripts/fetch_data.py`, "
            "or upload a file on the Data Ingestion page.",
        }
    except ValueError as exc:
        return pd.DataFrame(), {"status": "invalid", "message": str(exc)}
    if dataset.empty:
        return dataset, {"status": "invalid", "message": "Bundled files failed validation."}
    ingestion.save_processed(dataset)
    return dataset, {"status": "ready", "source": "raw", "n_reports": len(reports)}
