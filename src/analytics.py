"""Descriptive and diagnostic analytics: filters, KPIs, aggregations and correlations.

Every function is pure: it takes a prepared dataset (see :mod:`src.ingestion`) and
returns new values computed from it. Nothing here is hard-coded or cached.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FilterSpec:
    """Global filter selection shared by every page.

    Empty tuples mean "no restriction" for that attribute.

    Attributes:
        subjects: Subject codes to keep.
        schools: School codes to keep.
        sexes: Sex codes to keep.
        addresses: Address codes to keep.
        age_range: Inclusive ``(min, max)`` age, or ``None`` for all ages.
        exclude_g3_zero: Drop rows with G3 == 0 (likely non-appearance).
    """

    subjects: tuple[str, ...] = ()
    schools: tuple[str, ...] = ()
    sexes: tuple[str, ...] = ()
    addresses: tuple[str, ...] = ()
    age_range: tuple[int, int] | None = None
    exclude_g3_zero: bool = False


FILTER_COLUMNS: dict[str, str] = {
    "subjects": config.SUBJECT_COL,
    "schools": "school",
    "sexes": "sex",
    "addresses": "address",
}


def filter_options(df: pd.DataFrame) -> dict[str, object]:
    """Return the selectable values for each filter, derived from ``df``."""
    options: dict[str, object] = {
        name: sorted(df[col].dropna().astype(str).unique()) for name, col in FILTER_COLUMNS.items()
    }
    options["age_range"] = (int(df["age"].min()), int(df["age"].max()))
    return options


def apply_filters(df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    """Return the rows of ``df`` that satisfy every active filter in ``spec``."""
    mask = pd.Series(True, index=df.index)
    for name, col in FILTER_COLUMNS.items():
        selected = getattr(spec, name)
        if selected:
            mask &= df[col].astype(str).isin(selected)
    if spec.age_range is not None:
        lo, hi = spec.age_range
        mask &= df["age"].between(lo, hi)
    if spec.exclude_g3_zero:
        mask &= ~df[config.G3_ZERO_FLAG_COL].astype(bool)
    return df.loc[mask]


# ---------------------------------------------------------------------------
# KPIs and summaries
# ---------------------------------------------------------------------------
def kpis(df: pd.DataFrame) -> dict[str, float]:
    """Compute headline KPIs: count, mean/median G3, pass and fail rates.

    Returns NaN for rates and averages when ``df`` is empty.
    """
    n = len(df)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "mean_g3": nan, "median_g3": nan, "pass_rate": nan, "fail_rate": nan}
    pass_rate = float(df[config.PASS_COL].mean())
    return {
        "n": n,
        "mean_g3": float(df[config.TARGET_SOURCE].mean()),
        "median_g3": float(df[config.TARGET_SOURCE].median()),
        "pass_rate": pass_rate,
        "fail_rate": 1.0 - pass_rate,
    }


def _ordered_levels(series: pd.Series) -> list[object]:
    """Return a column's levels in natural order (categorical order, else sorted)."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return list(series.cat.categories)
    return sorted(series.dropna().unique().tolist())


def pass_rate_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Pass rate per level of ``by``.

    Returns:
        Columns ``[by, n, n_pass, pass_rate]`` in natural level order. Levels with
        no rows are kept with ``n == 0`` and NaN rate so axes stay comparable.
    """
    grouped = df.groupby(by, observed=False)[config.PASS_COL].agg(n="size", n_pass="sum")
    grouped = grouped.reindex(_ordered_levels(df[by]), fill_value=0)
    grouped["pass_rate"] = grouped["n_pass"] / grouped["n"].replace(0, np.nan)
    return grouped.rename_axis(by).reset_index()


def mean_by(df: pd.DataFrame, by: str, value: str = config.TARGET_SOURCE) -> pd.DataFrame:
    """Mean of ``value`` per level of ``by`` with size, std and standard error.

    Returns:
        Columns ``[by, n, mean, std, sem]`` in natural level order.
    """
    grouped = df.groupby(by, observed=False)[value].agg(n="size", mean="mean", std="std")
    grouped = grouped.reindex(_ordered_levels(df[by]))
    grouped["n"] = grouped["n"].fillna(0).astype(int)
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["n"].replace(0, np.nan))
    return grouped.rename_axis(by).reset_index()


def subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-subject summary: rows, mean/median G3, pass rate and G3 == 0 count."""
    g = df.groupby(config.SUBJECT_COL)
    out = pd.DataFrame(
        {
            "students": g.size(),
            "mean_g3": g[config.TARGET_SOURCE].mean(),
            "median_g3": g[config.TARGET_SOURCE].median(),
            "pass_rate": g[config.PASS_COL].mean(),
            "g3_zero": g[config.G3_ZERO_FLAG_COL].sum().astype(int),
        }
    )
    return out.rename_axis(config.SUBJECT_COL).reset_index()


def descriptive_stats(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Count, mean, std, min, quartiles and max for numeric columns.

    Args:
        df: Dataset.
        columns: Columns to describe; defaults to grades, age, absences, failures, studytime.
    """
    cols = columns or ["G1", "G2", "G3", "age", "absences", "failures", "studytime"]
    desc = df[cols].describe().T
    desc["count"] = desc["count"].astype(int)
    return desc.rename_axis("attribute").reset_index()


# ---------------------------------------------------------------------------
# Correlations and relationships
# ---------------------------------------------------------------------------
def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return schema numeric columns present in ``df`` with more than one distinct value."""
    return [c for c in config.NUMERIC_COLUMNS if c in df.columns and df[c].nunique() > 1]


def correlations_with(df: pd.DataFrame, target: str = config.TARGET_SOURCE) -> pd.Series:
    """Pearson correlation of every other numeric attribute with ``target``, sorted ascending."""
    cols = [c for c in numeric_feature_columns(df) if c != target]
    if len(df) < 3 or target not in df or df[target].nunique() < 2:
        return pd.Series(dtype=float)
    corr = df[cols].corrwith(df[target]).dropna()
    return corr.sort_values()


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of the numeric schema attributes."""
    return df[numeric_feature_columns(df)].corr()


def point_counts(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    """Return, for each row, how many rows share its exact ``(x, y)`` value pair."""
    return df.groupby([x, y])[x].transform("size")
