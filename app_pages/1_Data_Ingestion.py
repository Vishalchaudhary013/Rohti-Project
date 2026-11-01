"""Data Ingestion: upload CSV/XLSX files (or use the bundled dataset), validate, preview."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, ingestion, storage, ui

ui.setup_page("Data Ingestion")
ui.page_header(
    "Data Ingestion",
    "Upload one or more student files (CSV or XLSX). Each file is checked against the "
    "33-column schema; invalid rows are removed and reported, never silently changed.",
    eyebrow="Step 1",
)


@st.cache_data(show_spinner=False)
def _parse(data: bytes, name: str) -> tuple[pd.DataFrame | None, str]:
    """Parse uploaded bytes once per file content; return ``(table, error_message)``."""
    try:
        return ingestion.read_table(data, name), ""
    except ValueError as exc:
        return None, str(exc)


def _load(result: ingestion.UploadResult, files: dict[str, bytes], source: str) -> None:
    """Register uploads in SQLite, save the processed file and activate the dataset."""
    for name, report in result.reports.items():
        try:
            storage.register_upload(
                filename=name,
                row_count=report.n_rows,
                rows_kept=result.rows_kept.get(name, 0),
                checksum=ingestion.file_checksum(files[name]),
                status="accepted" if report.is_valid else "rejected",
                subject=result.subjects.get(name),
            )
        except Exception:  # history is best-effort; never block loading on it
            st.toast("Upload history could not be saved.", icon="⚠️")
    if not result.dataset.empty:
        ingestion.save_processed(result.dataset)
        ui.set_active_dataset(result.dataset, {"source": source, "files": list(result.reports)})
    st.session_state["last_result"] = result


# ---------------------------------------------------------------- upload panel
labels = config.SUBJECT_LABELS
codes = list(labels)

left, right = st.columns([2, 1], gap="large")
with left:
    ui.section(
        "Upload files", "Drop one file per subject, or a single file with a 'subject' column."
    )
    uploads = st.file_uploader(
        "Student data files",
        type=list(config.UPLOAD_TYPES),
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
with right:
    ui.section("Need a starting point?")
    st.download_button(
        "Download CSV template",
        ingestion.template_csv(),
        file_name="student_template.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )
    if ui.bundled_available():
        use_bundled = st.button(
            "Use bundled UCI dataset", icon=":material/dataset:", width="stretch"
        )
    else:
        use_bundled = False
        st.caption("The bundled UCI dataset is not included in this deployment.")

items: list[ingestion.UploadItem] = []
file_bytes: dict[str, bytes] = {}
parse_errors: list[str] = []
if uploads:
    ui.section("Assign subjects", "Detected from the file name; change it if needed.")
    for up in uploads:
        data = up.getvalue()
        table, error = _parse(data, up.name)
        if table is None:
            parse_errors.append(f"**{up.name}**: {error}")
            continue
        c1, c2, c3 = st.columns([3, 1, 2])
        c1.markdown(f"**{up.name}**")
        c2.caption(f"{len(table):,} rows · {up.size / 1024:,.0f} KB")
        if config.SUBJECT_COL in table.columns:
            c3.caption("Subject taken from the file's 'subject' column")
            subject = config.DEFAULT_UPLOAD_SUBJECT
        else:
            guess = ingestion.infer_subject(up.name)
            subject = c3.selectbox(
                f"Subject for {up.name}",
                codes,
                index=codes.index(guess),
                format_func=lambda c: labels[c],
                key=f"subject_{up.name}",
                label_visibility="collapsed",
            )
        items.append(ingestion.UploadItem(up.name, table, subject))
        file_bytes[up.name] = data
    for msg in parse_errors:
        ui.friendly_error(msg)
    if items and st.button("Validate and load", type="primary", icon=":material/check:"):
        with st.spinner("Validating…"):
            _load(ingestion.prepare_uploads(items), file_bytes, "upload")

if use_bundled:
    with st.spinner("Loading bundled dataset…"):
        try:
            raw = {p.name: p.read_bytes() for p in (config.RAW_MAT_FILE, config.RAW_POR_FILE)}
            bundled_items = [
                ingestion.UploadItem(n, ingestion.read_table(b, n), ingestion.infer_subject(n))
                for n, b in raw.items()
            ]
            _load(ingestion.prepare_uploads(bundled_items), raw, "bundled")
        except (OSError, ValueError) as exc:
            ui.friendly_error("The bundled dataset could not be loaded.", str(exc))

# ---------------------------------------------------------------- results
result: ingestion.UploadResult | None = st.session_state.get("last_result")
dataset, meta = ui.get_active_dataset()

if result is not None:
    ui.section("Validation results")
    accepted = sum(r.is_valid for r in result.reports.values())
    dropped = sum(r.n_invalid_rows for r in result.reports.values() if r.is_valid)
    if result.dataset.empty:
        st.markdown(ui.badge_html("No file passed validation", "critical"), unsafe_allow_html=True)
    else:
        st.markdown(ui.badge_html("Dataset loaded", "good"), unsafe_allow_html=True)
    ui.kpi_row(
        [
            ui.Kpi("Files accepted", f"{accepted} / {len(result.reports)}"),
            ui.Kpi("Rows loaded", f"{len(result.dataset):,}"),
            ui.Kpi("Rows removed", f"{dropped:,}", "Broke a schema rule"),
            ui.Kpi(
                "G3 = 0 rows",
                f"{int(result.dataset[config.G3_ZERO_FLAG_COL].sum()):,}"
                if not result.dataset.empty
                else "—",
                "Likely did not sit the final exam",
            ),
        ]
    )
    tabs = st.tabs(list(result.reports))
    for tab, report in zip(tabs, result.reports.values(), strict=True):
        with tab:
            kind = (
                "good"
                if report.is_valid and not report.count("error")
                else ("warning" if report.is_valid else "critical")
            )
            status = "Accepted" if report.is_valid else "Rejected"
            st.markdown(
                ui.badge_html(
                    f"{status} · {report.count('error')} errors · "
                    f"{report.count('warning')} warnings",
                    kind,
                ),
                unsafe_allow_html=True,
            )
            issues = report.to_frame()
            if issues.empty:
                st.caption("No issues found.")
            else:
                st.dataframe(issues, hide_index=True, width="stretch")

if dataset is not None:
    ui.section(
        "Data preview",
        f"Active dataset ({meta.get('source', 'upload')}): {len(dataset):,} rows. "
        "Showing the first 100.",
    )
    preview_tab, missing_tab = st.tabs(["Preview", "Missing values"])
    with preview_tab:
        st.dataframe(dataset.head(100), width="stretch", hide_index=True)
    with missing_tab:
        ms = ingestion.missing_summary(dataset)
        st.dataframe(
            ms,
            hide_index=True,
            width="stretch",
            column_config={
                "pct_missing": st.column_config.NumberColumn("% missing", format="%.1f%%")
            },
        )
elif result is None:
    ui.empty_state("Upload one or more files above to begin.")

# ---------------------------------------------------------------- history
ui.section("Upload history", "Every upload is logged with its row count and SHA-256 checksum.")
try:
    history = storage.list_uploads()
except Exception:  # never surface a stack trace in the UI
    history = pd.DataFrame()
    ui.friendly_error("Upload history is unavailable right now.")
if history.empty:
    st.caption("No uploads yet.")
else:
    history["checksum"] = history["checksum"].str[:12] + "…"
    st.dataframe(history, hide_index=True, width="stretch")
