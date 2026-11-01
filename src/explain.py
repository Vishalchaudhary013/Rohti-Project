"""Per-student risk tiers and human-readable reason codes.

Contributions are model-agnostic "what-if" effects: for each input feature the student's value
is replaced by a typical value from the reference data (median for numbers, most common value
for categories) and the change in predicted P(fail) is recorded:

    contribution = P(fail | actual values) − P(fail | this feature set to typical)

A positive contribution means the feature *raises* this student's risk. The three largest
positive contributions become the reason codes, phrased from the student's actual values and
where they sit in the reference distribution (e.g. "absences in top 10%").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config, modeling

TIERS: tuple[str, ...] = ("High", "Medium", "Low")
MIN_CONTRIBUTION = 0.01  # ignore effects smaller than one percentage point
NO_REASON = "No single strong driver"

_STUDYTIME = {1: "under 2 h", 2: "2–5 h", 3: "5–10 h", 4: "over 10 h"}
_EDU = {0: "none", 1: "primary (4th grade)", 2: "5th–9th grade", 3: "secondary", 4: "higher"}
_LIKERT_LOW = {
    "famrel": "poor family relationships",
    "health": "poor current health",
    "freetime": "little free time",
}
_LIKERT_HIGH = {
    "goout": "goes out with friends often",
    "Dalc": "high workday alcohol use",
    "Walc": "high weekend alcohol use",
    "freetime": "a lot of free time",
}
_YES_PHRASES = {
    "schoolsup": "receives extra school support",
    "famsup": "family educational support",
    "paid": "takes extra paid classes",
    "activities": "extra-curricular activities",
    "nursery": "attended nursery school",
    "higher": "plans higher education",
    "internet": "internet access at home",
    "romantic": "in a romantic relationship",
}
_NO_PHRASES = {
    "schoolsup": "no extra school support",
    "famsup": "no family educational support",
    "paid": "no extra paid classes",
    "activities": "no extra-curricular activities",
    "nursery": "did not attend nursery school",
    "higher": "does not plan higher education",
    "internet": "no internet access at home",
    "romantic": "not in a relationship",
}


_CATEGORY_PHRASES: dict[str, dict[str, str]] = {
    "Pstatus": {"A": "parents live apart", "T": "parents live together"},
    "famsize": {"LE3": "family of three or fewer", "GT3": "family of more than three"},
    "address": {"U": "lives in an urban area", "R": "lives in a rural area"},
    "sex": {"F": "female", "M": "male"},
    "school": {"GP": "attends Gabriel Pereira", "MS": "attends Mousinho da Silveira"},
    "guardian": {
        "mother": "guardian: mother",
        "father": "guardian: father",
        "other": "guardian is not a parent",
    },
    "reason": {
        "home": "chose school for proximity to home",
        "reputation": "chose school for its reputation",
        "course": "chose school for its courses",
        "other": "chose school for other reasons",
    },
}


# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------
def risk_tier(p_fail: float) -> str:
    """Map P(fail) to a tier: High ≥ 0.7, Medium 0.4–0.7, Low < 0.4."""
    if p_fail >= config.RISK_HIGH:
        return "High"
    if p_fail >= config.RISK_MEDIUM:
        return "Medium"
    return "Low"


def risk_tiers(p_fail: pd.Series | np.ndarray) -> pd.Categorical:
    """Vectorised :func:`risk_tier` as an ordered categorical (High > Medium > Low)."""
    labels = [risk_tier(float(p)) for p in np.asarray(p_fail)]
    return pd.Categorical(labels, categories=list(TIERS), ordered=True)


def tier_counts(tiers: pd.Series | pd.Categorical) -> dict[str, int]:
    """Count students per tier, always returning all three tiers."""
    counts = pd.Series(tiers).value_counts()
    return {t: int(counts.get(t, 0)) for t in TIERS}


# ---------------------------------------------------------------------------
# Contributions
# ---------------------------------------------------------------------------
def model_inputs(pipe: Pipeline) -> list[str]:
    """Raw input columns the fitted pipeline uses."""
    numeric, categorical = modeling.feature_columns_from(pipe)
    return numeric + categorical


def typical_values(reference: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    """Median of numeric and most common value of categorical columns in ``reference``."""
    out: dict[str, object] = {}
    for col in columns:
        if col in config.NUMERIC_COLUMNS:
            out[col] = float(reference[col].median())
        else:
            out[col] = reference[col].mode(dropna=True).iloc[0]
    return out


def contributions(pipe: Pipeline, df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """What-if contribution of each input feature to each row's P(fail).

    Args:
        pipe: Fitted pipeline (see :mod:`src.modeling`).
        df: Students to explain.
        reference: Data that defines "typical" values (usually the full dataset).

    Returns:
        DataFrame aligned to ``df.index`` with one column per input feature.
    """
    cols = model_inputs(pipe)
    typical = typical_values(reference, cols)
    base = modeling.predict_fail_proba(pipe, df)
    out = {}
    for col in cols:
        swapped = df.copy()
        swapped[col] = typical[col]
        out[col] = base - modeling.predict_fail_proba(pipe, swapped)
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# Wording
# ---------------------------------------------------------------------------
def _percentile(reference: pd.Series, value: float) -> float:
    """Share of reference values strictly below ``value`` (0–1)."""
    return float((reference < value).mean())


def describe(feature: str, value: object, reference: pd.DataFrame) -> str:
    """Turn one feature value into a short, human-readable reason."""
    if feature == "failures":
        n = int(value)
        return f"{n}{'+' if n >= 3 else ''} past failure{'s' if n != 1 else ''}"
    if feature == "absences":
        pct = _percentile(reference["absences"], float(value))
        if pct >= 0.9:
            return f"absences in top 10% ({int(value)})"
        if pct >= 0.75:
            return f"absences in top 25% ({int(value)})"
        return f"{int(value)} absences"
    if feature in ("G1", "G2"):
        period = "first" if feature == "G1" else "second"
        tag = " (below pass mark)" if float(value) < config.PASS_THRESHOLD else ""
        return f"{period}-period grade {int(value)}/20{tag}"
    if feature == "studytime":
        return f"studies {_STUDYTIME.get(int(value), value)} a week"
    if feature in ("Medu", "Fedu"):
        who = "mother's" if feature == "Medu" else "father's"
        return f"{who} education: {_EDU.get(int(value), value)}"
    if feature == "age":
        older = float(value) > float(reference["age"].median())
        return f"age {int(value)}{' (older than most peers)' if older else ''}"
    if feature == "traveltime":
        return "long journey to school" if int(value) >= 3 else "short journey to school"
    if feature in _LIKERT_HIGH and int(value) >= 4:
        return f"{_LIKERT_HIGH[feature]} ({int(value)}/5)"
    if feature in _LIKERT_LOW and int(value) <= 2:
        return f"{_LIKERT_LOW[feature]} ({int(value)}/5)"
    if feature in _YES_PHRASES:
        return _YES_PHRASES[feature] if value == "yes" else _NO_PHRASES[feature]
    if feature in _CATEGORY_PHRASES:
        return _CATEGORY_PHRASES[feature].get(str(value), f"{feature}: {value}")
    if feature in ("Mjob", "Fjob"):
        who = "mother's" if feature == "Mjob" else "father's"
        return f"{who} job: {str(value).replace('_', ' ')}"
    if feature == config.SUBJECT_COL:
        return f"subject: {config.SUBJECT_LABELS.get(str(value), value)}"
    spec = config.SCHEMA.get(feature)
    label = spec.description if spec else feature
    shown = config.VALUE_LABELS.get(feature, {}).get(value, value)
    return f"{label}: {shown}"


def reason_codes(
    pipe: Pipeline, df: pd.DataFrame, reference: pd.DataFrame, top_n: int = config.TOP_N_REASONS
) -> pd.DataFrame:
    """Top-``top_n`` reasons per student, from the largest positive contributions.

    Returns:
        DataFrame aligned to ``df.index`` with columns ``reason_1`` … ``reason_n`` (empty
        string when fewer reasons qualify), ``reasons`` (joined text) and ``top_features``.
    """
    contrib = contributions(pipe, df, reference)
    rows = []
    for idx, row in contrib.iterrows():
        top = row[row > MIN_CONTRIBUTION].sort_values(ascending=False).head(top_n)
        texts = [describe(f, df.at[idx, f], reference) for f in top.index]
        padded = texts + [""] * (top_n - len(texts))
        rows.append(
            {
                **{f"reason_{i + 1}": t for i, t in enumerate(padded)},
                "reasons": "; ".join(texts) if texts else NO_REASON,
                "top_features": ", ".join(top.index),
            }
        )
    return pd.DataFrame(rows, index=df.index)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
DISPLAY_COLUMNS: tuple[str, ...] = (
    config.RECORD_ID_COL,
    config.SUBJECT_COL,
    "school",
    "sex",
    "age",
    "failures",
    "absences",
)


def score(
    pipe: Pipeline,
    df: pd.DataFrame,
    threshold: float = config.DEFAULT_THRESHOLD,
    reference: pd.DataFrame | None = None,
    explain_all: bool = False,
) -> pd.DataFrame:
    """Score every student: P(fail), tier, at-risk flag and reason codes.

    Args:
        pipe: Fitted pipeline.
        df: Students to score.
        threshold: Flag a student when P(fail) ≥ threshold.
        reference: Data defining typical values for reasons (defaults to ``df``).
        explain_all: Compute reasons for every row, not only flagged students.

    Returns:
        One row per student, sorted by descending P(fail).
    """
    reference = df if reference is None else reference
    p_fail = modeling.predict_fail_proba(pipe, df)
    out = df[[c for c in DISPLAY_COLUMNS if c in df.columns]].copy()
    out["p_fail"] = p_fail
    out["risk_tier"] = risk_tiers(p_fail)
    out["predicted_fail"] = p_fail >= threshold
    targets = df if explain_all else df[out["predicted_fail"].to_numpy()]
    reasons = (
        reason_codes(pipe, targets, reference)
        if len(targets)
        else pd.DataFrame(columns=["reason_1", "reason_2", "reason_3", "reasons", "top_features"])
    )
    out = out.join(reasons)
    for col in reasons.columns:
        out[col] = out[col].fillna("")
    return out.sort_values("p_fail", ascending=False)


def filter_scored(
    scored: pd.DataFrame, tiers: tuple[str, ...] = (), query: str = "", only_flagged: bool = True
) -> pd.DataFrame:
    """Filter a scored table by tier, at-risk flag and a case-insensitive text search."""
    out = scored
    if only_flagged:
        out = out[out["predicted_fail"]]
    if tiers:
        out = out[out["risk_tier"].astype(str).isin(tiers)]
    q = query.strip().lower()
    if q:
        text = out.astype(str).agg(" ".join, axis=1).str.lower()
        out = out[text.str.contains(q, regex=False)]
    return out
