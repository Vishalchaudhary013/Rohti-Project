"""Data ingestion: loading, schema validation, cleaning, merging and derived columns.

Validation never raises on bad data. It returns a :class:`ValidationReport` that the UI
and CLI render, and :func:`clean` drops (and counts) rows that break hard schema rules.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
import pandas as pd

from src import config

Severity = Literal["error", "warning", "info"]


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationIssue:
    """A single finding from schema validation.

    Attributes:
        severity: ``"error"`` (rows/columns unusable), ``"warning"`` or ``"info"``.
        check: Short machine-readable name of the check, e.g. ``"range"``.
        column: Affected column, or ``None`` for table-level findings.
        message: Human-readable explanation.
        n_rows: Number of affected rows (0 for column-level findings).
    """

    severity: Severity
    check: str
    column: str | None
    message: str
    n_rows: int = 0


@dataclass
class ValidationReport:
    """Structured result of validating one table against :data:`config.SCHEMA`.

    Attributes:
        source: Name of the validated file or table.
        n_rows: Row count of the raw input.
        n_columns: Column count of the raw input.
        issues: All findings, in the order they were detected.
        invalid_row_mask: Boolean mask of rows breaking a hard rule (index-aligned).
    """

    source: str
    n_rows: int
    n_columns: int
    issues: list[ValidationIssue] = field(default_factory=list)
    invalid_row_mask: pd.Series | None = None

    @property
    def is_valid(self) -> bool:
        """Return True when there are no table-level (column) errors.

        Row-level errors do not make a table unusable; :func:`clean` drops those rows.
        """
        return not any(i.severity == "error" and i.n_rows == 0 for i in self.issues)

    @property
    def n_invalid_rows(self) -> int:
        """Return the number of rows that break at least one hard rule."""
        if self.invalid_row_mask is None:
            return 0
        return int(self.invalid_row_mask.sum())

    def count(self, severity: Severity) -> int:
        """Return the number of issues with the given severity."""
        return sum(1 for i in self.issues if i.severity == severity)

    def to_frame(self) -> pd.DataFrame:
        """Return the issues as a DataFrame suitable for display."""
        columns = ["severity", "check", "column", "message", "n_rows"]
        return pd.DataFrame([vars(i) for i in self.issues], columns=columns)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _sniff_separator(sample: str) -> str:
    """Pick ``;`` or ``,`` as CSV separator from the header line of ``sample``."""
    header = sample.splitlines()[0] if sample else ""
    return ";" if header.count(";") >= header.count(",") else ","


def read_table(source: Path | BinaryIO | bytes, filename: str | None = None) -> pd.DataFrame:
    """Read a CSV (``;`` or ``,`` delimited) or XLSX file into a DataFrame.

    Args:
        source: A path, raw bytes, or a binary file-like object (e.g. a Streamlit upload).
        filename: Name used to detect the format when ``source`` is not a path.

    Returns:
        The raw table with whitespace-stripped column names.

    Raises:
        ValueError: If the format is unsupported or the file cannot be parsed.
    """
    if isinstance(source, Path):
        name = source.name
        data = source.read_bytes()
    else:
        name = filename or getattr(source, "name", "") or ""
        data = source if isinstance(source, bytes) else source.read()

    suffix = Path(name).suffix.lower().lstrip(".")
    if suffix not in config.UPLOAD_TYPES:
        raise ValueError(
            f"Unsupported file type '.{suffix}'. Upload one of: "
            + ", ".join(f".{t}" for t in config.UPLOAD_TYPES)
        )
    try:
        if suffix == "xlsx":
            df = pd.read_excel(io.BytesIO(data), engine="openpyxl")
        else:
            text = data.decode("utf-8-sig")
            df = pd.read_csv(io.StringIO(text), sep=_sniff_separator(text))
    except Exception as exc:  # any parser error becomes a clean, user-facing message
        raise ValueError(f"Could not parse '{name}': {exc}") from exc

    df.columns = [str(c).strip() for c in df.columns]
    return df


def file_checksum(data: bytes | Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    raw = data.read_bytes() if isinstance(data, Path) else data
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing-value counts and percentages, worst first."""
    n = len(df)
    missing = df.isna().sum()
    out = pd.DataFrame(
        {
            "column": missing.index,
            "n_missing": missing.to_numpy(dtype=int),
            "pct_missing": (missing.to_numpy() / n * 100) if n else np.zeros(len(missing)),
        }
    )
    return out.sort_values(["n_missing", "column"], ascending=[False, True]).reset_index(drop=True)


def _normalise_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding whitespace and quotes from string cells."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: v.strip().strip('"').strip() if isinstance(v, str) else v
            )
    return out


def validate(df: pd.DataFrame, source: str = "table") -> ValidationReport:
    """Validate a raw table against the 33-attribute UCI schema.

    Checks performed: non-empty table, required columns present, unexpected columns,
    missing values, numeric dtype, integer values, numeric ranges, categorical domains,
    exact duplicate rows, and G3 == 0 (likely non-appearance).

    Args:
        df: Raw table as read by :func:`read_table`.
        source: Name of the file, used in the report.

    Returns:
        A :class:`ValidationReport`; never raises for data problems.
    """
    report = ValidationReport(source=source, n_rows=len(df), n_columns=df.shape[1])
    add = report.issues.append

    if df.empty:
        add(ValidationIssue("error", "empty", None, "The table has no rows."))
        report.invalid_row_mask = pd.Series(False, index=df.index)
        return report

    df = _normalise_strings(df)
    missing_cols = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    for col in missing_cols:
        add(
            ValidationIssue("error", "required_column", col, f"Required column '{col}' is missing.")
        )
    allowed_extra = set(config.DERIVED_COLUMNS)
    for col in df.columns:
        if col not in config.SCHEMA and col not in allowed_extra:
            add(ValidationIssue("info", "extra_column", col, f"Column '{col}' is not used."))

    invalid = pd.Series(False, index=df.index)
    for col, spec in config.SCHEMA.items():
        if col not in df.columns:
            continue
        series = df[col]
        na = series.isna()
        if na.any():
            add(
                ValidationIssue(
                    "error", "missing", col, f"{int(na.sum())} missing value(s).", int(na.sum())
                )
            )
            invalid |= na

        if spec.kind == "numeric":
            numeric = pd.to_numeric(series, errors="coerce")
            non_numeric = numeric.isna() & ~na
            if non_numeric.any():
                n = int(non_numeric.sum())
                add(ValidationIssue("error", "dtype", col, f"{n} non-numeric value(s).", n))
                invalid |= non_numeric
            valid = numeric.notna()
            non_integer = valid & (numeric % 1 != 0)
            if non_integer.any():
                n = int(non_integer.sum())
                add(ValidationIssue("error", "integer", col, f"{n} non-integer value(s).", n))
                invalid |= non_integer
            out_of_range = valid & ((numeric < spec.min_value) | (numeric > spec.max_value))
            if out_of_range.any():
                n = int(out_of_range.sum())
                add(
                    ValidationIssue(
                        "error",
                        "range",
                        col,
                        f"{n} value(s) outside [{spec.min_value}, {spec.max_value}].",
                        n,
                    )
                )
                invalid |= out_of_range
        else:
            assert spec.allowed is not None
            bad = ~na & ~series.astype(str).isin(spec.allowed)
            if bad.any():
                n = int(bad.sum())
                examples = ", ".join(sorted(series[bad].astype(str).unique())[:3])
                add(
                    ValidationIssue(
                        "error",
                        "domain",
                        col,
                        f"{n} value(s) not in {list(spec.allowed)} (e.g. {examples}).",
                        n,
                    )
                )
                invalid |= bad

    dup = df.duplicated()
    if dup.any():
        n = int(dup.sum())
        add(
            ValidationIssue(
                "warning", "duplicate_rows", None, f"{n} exact duplicate row(s) (kept).", n
            )
        )

    if config.TARGET_SOURCE in df.columns:
        g3 = pd.to_numeric(df[config.TARGET_SOURCE], errors="coerce")
        zeros = int((g3 == 0).sum())
        if zeros:
            add(
                ValidationIssue(
                    "warning",
                    "g3_zero",
                    config.TARGET_SOURCE,
                    f"{zeros} row(s) with G3 = 0 (likely did not sit the final exam).",
                    zeros,
                )
            )

    report.invalid_row_mask = invalid
    return report


# ---------------------------------------------------------------------------
# Cleaning, merging and derived columns
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """Return a typed copy of ``df`` with rows that break hard rules removed.

    Args:
        df: Raw table that was passed to :func:`validate`.
        report: The report produced by :func:`validate` for ``df``.

    Returns:
        Table restricted to schema columns (plus a ``subject`` column if present),
        numeric columns cast to ``int64`` and categoricals to ``str``.

    Raises:
        ValueError: If the report contains table-level errors (e.g. missing columns).
    """
    if not report.is_valid:
        raise ValueError("Table has schema errors; fix the reported issues before cleaning.")
    out = _normalise_strings(df)
    if report.invalid_row_mask is not None:
        out = out.loc[~report.invalid_row_mask]
    keep = list(config.REQUIRED_COLUMNS)
    if config.SUBJECT_COL in out.columns:
        keep.append(config.SUBJECT_COL)
    out = out[keep].copy()
    for col in config.NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col]).astype("int64")
    for col in config.CATEGORICAL_COLUMNS:
        out[col] = out[col].astype(str)
    return out.reset_index(drop=True)


def merge_subjects(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-subject tables and add a ``subject`` column.

    Args:
        frames: Mapping of subject code (e.g. ``"mat"``) to cleaned table.

    Returns:
        The concatenated table with ``subject`` as the first column.
    """
    parts = []
    for subject, frame in frames.items():
        part = frame.drop(columns=[config.SUBJECT_COL], errors="ignore").copy()
        part.insert(0, config.SUBJECT_COL, subject)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def absence_band(absences: pd.Series) -> pd.Series:
    """Bin absence counts into the ordered bands 0, 1–5, 6–10, 11–20, 20+."""
    return pd.cut(
        absences,
        bins=list(config.ABSENCE_BAND_EDGES),
        labels=list(config.ABSENCE_BAND_LABELS),
        ordered=True,
    )


def flag_cross_subject_students(df: pd.DataFrame) -> pd.Series:
    """Flag rows whose student-matching attributes appear in more than one subject.

    Uses the 13 attributes the original authors used to join the two files. These are
    not a true student ID, so distinct students who share all 13 values are also matched;
    the flag is therefore an approximation and is reported as a known limitation.

    Args:
        df: Merged table containing a ``subject`` column.

    Returns:
        Boolean Series aligned to ``df.index``.
    """
    if config.SUBJECT_COL not in df.columns or df[config.SUBJECT_COL].nunique() < 2:
        return pd.Series(False, index=df.index)
    keys = list(config.STUDENT_MATCH_KEYS)
    n_subjects = df.groupby(keys, dropna=False)[config.SUBJECT_COL].transform("nunique")
    return (n_subjects > 1).astype(bool)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``pass``, ``absence_band`` and data-quality flag columns.

    Args:
        df: Cleaned (typed) table with a ``subject`` column.

    Returns:
        A copy of ``df`` with the derived columns appended.
    """
    out = df.copy()
    position = out.groupby(config.SUBJECT_COL, sort=False).cumcount() + 1
    out.insert(
        0,
        config.RECORD_ID_COL,
        out[config.SUBJECT_COL].astype(str) + "-" + position.map("{:04d}".format),
    )
    out[config.PASS_COL] = (out[config.TARGET_SOURCE] >= config.PASS_THRESHOLD).astype("int64")
    out[config.ABSENCE_BAND_COL] = absence_band(out["absences"])
    out[config.G3_ZERO_FLAG_COL] = out[config.TARGET_SOURCE] == 0
    out[config.BOTH_SUBJECTS_FLAG_COL] = flag_cross_subject_students(out)
    return out


def prepare(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, ValidationReport]]:
    """Validate, clean, merge and enrich one or more raw subject tables.

    Tables with table-level errors are excluded from the result; their reports explain why.

    Args:
        frames: Mapping of subject code (or source name) to raw table.

    Returns:
        ``(dataset, reports)`` where ``dataset`` is empty if no table was usable.
    """
    reports: dict[str, ValidationReport] = {}
    cleaned: dict[str, pd.DataFrame] = {}
    for subject, raw in frames.items():
        report = validate(raw, source=subject)
        reports[subject] = report
        if report.is_valid:
            cleaned[subject] = clean(raw, report)
    if not cleaned:
        return pd.DataFrame(), reports
    return add_derived_columns(merge_subjects(cleaned)), reports


def load_bundled(
    raw_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, ValidationReport]]:
    """Load and prepare the bundled UCI files from ``raw_dir``.

    Args:
        raw_dir: Directory containing ``student-mat.csv`` and ``student-por.csv``
            (defaults to ``config.RAW_DIR``).

    Returns:
        ``(dataset, reports)`` as for :func:`prepare`.

    Raises:
        FileNotFoundError: If either raw file is missing (run ``scripts/fetch_data.py``).
    """
    raw_dir = raw_dir or config.RAW_DIR
    paths = {"mat": raw_dir / config.RAW_MAT_FILE.name, "por": raw_dir / config.RAW_POR_FILE.name}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw file(s) {missing} in {raw_dir}. Run: python scripts/fetch_data.py"
        )
    return prepare({s: read_table(p) for s, p in paths.items()})


def save_processed(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the prepared dataset to CSV (default ``config.PROCESSED_FILE``); return the path."""
    path = path or config.PROCESSED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_processed(path: Path | None = None) -> pd.DataFrame:
    """Read a dataset written by :func:`save_processed`, restoring derived dtypes."""
    path = path or config.PROCESSED_FILE
    df = pd.read_csv(path)
    df[config.ABSENCE_BAND_COL] = pd.Categorical(
        df[config.ABSENCE_BAND_COL].astype(str),
        categories=list(config.ABSENCE_BAND_LABELS),
        ordered=True,
    )
    for col in (config.G3_ZERO_FLAG_COL, config.BOTH_SUBJECTS_FLAG_COL):
        df[col] = df[col].astype(bool)
    return df


# ---------------------------------------------------------------------------
# Data-quality summary
# ---------------------------------------------------------------------------
def data_quality_summary(df: pd.DataFrame) -> dict[str, object]:
    """Compute data-quality facts about a prepared dataset.

    Args:
        df: Output of :func:`prepare` / :func:`load_bundled`.

    Returns:
        Dictionary with total rows, per-subject breakdown (rows, G3 == 0, pass rate,
        cross-subject matches), approximate unique students, missing cells and
        absence-band counts. All values are computed from ``df``.
    """
    subj = config.SUBJECT_COL
    per_subject = (
        df.groupby(subj, sort=True)
        .agg(
            rows=(config.PASS_COL, "size"),
            g3_zero=(config.G3_ZERO_FLAG_COL, "sum"),
            pass_rate=(config.PASS_COL, "mean"),
            in_both=(config.BOTH_SUBJECTS_FLAG_COL, "sum"),
        )
        .reset_index()
    )
    matched_keys = df.loc[df[config.BOTH_SUBJECTS_FLAG_COL], list(config.STUDENT_MATCH_KEYS)]
    n_matched_keys = len(matched_keys.drop_duplicates())
    bands = df[config.ABSENCE_BAND_COL].value_counts(sort=False)
    return {
        "total_rows": len(df),
        "per_subject": per_subject,
        "g3_zero_total": int(df[config.G3_ZERO_FLAG_COL].sum()),
        "pass_rate_all": float(df[config.PASS_COL].mean()),
        "pass_rate_excl_g3_zero": float(
            df.loc[~df[config.G3_ZERO_FLAG_COL], config.PASS_COL].mean()
        )
        if (~df[config.G3_ZERO_FLAG_COL]).any()
        else float("nan"),
        "rows_in_both_subjects": int(df[config.BOTH_SUBJECTS_FLAG_COL].sum()),
        "matched_student_keys": n_matched_keys,
        "missing_cells": int(df.isna().sum().sum()),
        "absence_bands": {str(k): int(v) for k, v in bands.items()},
    }


def format_quality_summary(
    summary: Mapping[str, object], reports: Mapping[str, ValidationReport]
) -> str:
    """Render the data-quality summary and validation reports as plain text."""
    lines = ["=" * 64, "DATA-QUALITY SUMMARY", "=" * 64]
    for name, rep in reports.items():
        status = "OK" if rep.is_valid else "REJECTED"
        lines.append(
            f"[{status}] {name}: {rep.n_rows} rows x {rep.n_columns} cols | "
            f"errors={rep.count('error')} warnings={rep.count('warning')} "
            f"dropped_rows={rep.n_invalid_rows}"
        )
        for issue in rep.issues:
            col = f"{issue.column}: " if issue.column else ""
            lines.append(f"    - {issue.severity.upper():7} {col}{issue.message}")
    lines.append("-" * 64)
    lines.append(f"Total rows (merged): {summary['total_rows']}")
    per_subject = summary["per_subject"]
    assert isinstance(per_subject, pd.DataFrame)
    for row in per_subject.itertuples(index=False):
        label = config.SUBJECT_LABELS.get(str(row.subject), str(row.subject))
        lines.append(
            f"  {label:<12} rows={row.rows:<4} G3=0: {int(row.g3_zero):<3} "
            f"pass rate={row.pass_rate:.1%}  in both subjects={int(row.in_both)}"
        )
    lines += [
        f"G3 = 0 rows (likely non-appearance): {summary['g3_zero_total']}",
        f"Pass rate, all rows: {summary['pass_rate_all']:.1%}",
        f"Pass rate, excluding G3 = 0: {summary['pass_rate_excl_g3_zero']:.1%}",
        f"Rows matched in both subject files: {summary['rows_in_both_subjects']} "
        f"({summary['matched_student_keys']} distinct matching-key combinations)",
        f"Missing cells after cleaning: {summary['missing_cells']}",
        "Absence bands: "
        + ", ".join(f"{k}: {v}" for k, v in dict(summary["absence_bands"]).items()),  # type: ignore[call-overload]
        "=" * 64,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-file uploads
# ---------------------------------------------------------------------------
def infer_subject(filename: str) -> str:
    """Guess the subject code from a file name (``student-mat.csv`` -> ``"mat"``).

    Returns:
        ``"mat"``, ``"por"`` or :data:`config.DEFAULT_UPLOAD_SUBJECT` when unknown.
    """
    stem = Path(filename).stem.lower()
    for code, label in config.SUBJECT_LABELS.items():
        if code == config.DEFAULT_UPLOAD_SUBJECT:
            continue
        if code in stem.replace("_", "-").split("-") or label.lower() in stem or stem == code:
            return code
    if "math" in stem:
        return "mat"
    if "portug" in stem:
        return "por"
    return config.DEFAULT_UPLOAD_SUBJECT


@dataclass(frozen=True)
class UploadItem:
    """One parsed upload awaiting validation.

    Attributes:
        name: Original file name.
        table: Raw table as returned by :func:`read_table`.
        subject: Subject code to assign; ignored if the table has a ``subject`` column.
    """

    name: str
    table: pd.DataFrame
    subject: str


@dataclass
class UploadResult:
    """Outcome of preparing one or more uploads.

    Attributes:
        dataset: Merged, cleaned and enriched table (empty if nothing was usable).
        reports: Validation report per file name.
        rows_kept: Rows kept per file name after cleaning.
        subjects: Subject code(s) each file contributed.
    """

    dataset: pd.DataFrame
    reports: dict[str, ValidationReport]
    rows_kept: dict[str, int]
    subjects: dict[str, str]


def prepare_uploads(items: list[UploadItem]) -> UploadResult:
    """Validate, clean and merge several uploaded files into one dataset.

    Files with a ``subject`` column keep it (split per value); other files take the
    subject chosen by the user. Rejected files are reported and skipped. Several files
    for the same subject are stacked.

    Args:
        items: Parsed uploads.

    Returns:
        An :class:`UploadResult`; ``dataset`` is empty if every file was rejected.
    """
    reports: dict[str, ValidationReport] = {}
    rows_kept: dict[str, int] = {}
    subjects: dict[str, str] = {}
    by_subject: dict[str, list[pd.DataFrame]] = {}
    for item in items:
        report = validate(item.table, source=item.name)
        reports[item.name] = report
        if not report.is_valid:
            rows_kept[item.name] = 0
            subjects[item.name] = "—"
            continue
        cleaned = clean(item.table, report)
        rows_kept[item.name] = len(cleaned)
        if config.SUBJECT_COL in cleaned.columns:
            groups = {str(s): g for s, g in cleaned.groupby(config.SUBJECT_COL, sort=False)}
        else:
            groups = {item.subject: cleaned}
        subjects[item.name] = ", ".join(groups)
        for subject, frame in groups.items():
            by_subject.setdefault(subject, []).append(frame)
    if not by_subject:
        return UploadResult(pd.DataFrame(), reports, rows_kept, subjects)
    frames = {s: pd.concat(parts, ignore_index=True) for s, parts in by_subject.items()}
    dataset = add_derived_columns(merge_subjects(frames))
    return UploadResult(dataset, reports, rows_kept, subjects)


def template_csv() -> str:
    """Return a header-only CSV template (``;``-delimited) with every required column."""
    return config.CSV_SEPARATOR.join(config.REQUIRED_COLUMNS) + "\n"
