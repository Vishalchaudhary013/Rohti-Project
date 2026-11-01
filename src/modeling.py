"""Modelling: preprocessing pipelines, cross-validation, metrics, persistence and inference.

Target: ``pass = 1 if G3 >= 10``. The early-warning task flags *fail* (``pass == 0``), so
fail-class recall and precision are reported alongside the usual pass-class metrics.

Validation: 5-fold ``StratifiedKFold(shuffle=True, random_state=42)``. Reported metrics are the
mean ± std across folds at the default 0.5 threshold; out-of-fold probabilities are kept so the
UI can recompute confusion matrices, ROC curves and threshold trade-offs without refitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config

Setting = Literal["A", "B"]
ClassWeight = Literal["balanced"] | None

MODEL_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}
BASELINE = "baseline"
BASELINE_LABEL = "Majority-class baseline"
METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision": "Precision (pass)",
    "recall": "Recall (pass)",
    "f1": "F1 (pass)",
    "roc_auc": "ROC-AUC",
    "fail_recall": "Fail-class recall",
    "fail_precision": "Fail-class precision",
}
MIN_ROWS_PER_CLASS = config.CV_FOLDS


# ---------------------------------------------------------------------------
# Features and pipelines
# ---------------------------------------------------------------------------
def feature_columns(df: pd.DataFrame, setting: Setting) -> tuple[list[str], list[str]]:
    """Return ``(numeric, categorical)`` model inputs for a feature setting.

    Setting A uses all 32 non-target attributes (including G1, G2); setting B drops G1 and G2.
    ``subject`` is added as a categorical input when the data spans more than one subject.
    """
    excluded = set(config.NON_FEATURE_COLUMNS)
    if setting == "B":
        excluded |= set(config.SETTING_B_EXCLUDED)
    numeric = [c for c in config.NUMERIC_COLUMNS if c not in excluded]
    categorical = [c for c in config.CATEGORICAL_COLUMNS if c not in excluded]
    if config.SUBJECT_COL in df.columns and df[config.SUBJECT_COL].nunique() > 1:
        categorical.append(config.SUBJECT_COL)
    return numeric, categorical


def make_estimator(name: str, class_weight: ClassWeight = None):
    """Create an unfitted classifier with the project's fixed hyper-parameters."""
    rs = config.RANDOM_STATE
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=config.LOGREG_MAX_ITER, class_weight=class_weight, random_state=rs
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=config.DT_MAX_DEPTH, class_weight=class_weight, random_state=rs
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            class_weight=class_weight,
            random_state=rs,
            n_jobs=-1,
        )
    if name == BASELINE:
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"Unknown model '{name}'")


def build_pipeline(
    df: pd.DataFrame, setting: Setting, name: str, class_weight: ClassWeight = None
) -> Pipeline:
    """Preprocessing (scale numerics, one-hot categoricals) followed by the classifier."""
    numeric, categorical = feature_columns(df, setting)
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("model", make_estimator(name, class_weight))])


def target(df: pd.DataFrame) -> np.ndarray:
    """Return the binary target (1 = pass)."""
    return df[config.PASS_COL].to_numpy(dtype=int)


def can_model(df: pd.DataFrame) -> tuple[bool, str]:
    """Check there are enough rows of each class for stratified cross-validation."""
    if config.PASS_COL not in df or df.empty:
        return False, "No data loaded."
    counts = df[config.PASS_COL].value_counts()
    smallest = int(counts.min()) if len(counts) == 2 else 0
    if smallest < MIN_ROWS_PER_CLASS:
        return False, (
            f"Need at least {MIN_ROWS_PER_CLASS} passing and {MIN_ROWS_PER_CLASS} failing "
            f"students for {config.CV_FOLDS}-fold cross-validation (smallest class: {smallest})."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics_at(y: np.ndarray, proba_pass: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute all metrics from true labels and P(pass), flagging fail when P(fail) ≥ threshold.

    Args:
        y: True labels (1 = pass).
        proba_pass: Predicted probability of passing.
        threshold: Decision threshold on P(fail); default 0.5.
    """
    pred = np.where(1 - proba_pass >= threshold, 0, 1)
    auc = roc_auc_score(y, proba_pass) if len(np.unique(y)) == 2 else float("nan")
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, pos_label=1, zero_division=0),
        "recall": recall_score(y, pred, pos_label=1, zero_division=0),
        "f1": f1_score(y, pred, pos_label=1, zero_division=0),
        "roc_auc": auc,
        "fail_recall": recall_score(y, pred, pos_label=0, zero_division=0),
        "fail_precision": precision_score(y, pred, pos_label=0, zero_division=0),
    }


def confusion(y: np.ndarray, proba_pass: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Confusion matrix with rows = actual [Fail, Pass] and columns = predicted [Fail, Pass]."""
    pred = np.where(1 - proba_pass >= threshold, 0, 1)
    return confusion_matrix(y, pred, labels=[0, 1])


def threshold_curve(y: np.ndarray, proba_pass: np.ndarray, steps: int = 91) -> pd.DataFrame:
    """Fail-class recall and precision across thresholds 0.05–0.95 on P(fail)."""
    rows = []
    for t in np.linspace(0.05, 0.95, steps):
        m = metrics_at(y, proba_pass, float(t))
        rows.append(
            {
                "threshold": float(t),
                "fail_recall": m["fail_recall"],
                "fail_precision": m["fail_precision"],
                "flagged": float((1 - proba_pass >= t).mean()),
            }
        )
    return pd.DataFrame(rows)


def roc_points(y: np.ndarray, proba_pass: np.ndarray) -> pd.DataFrame:
    """ROC curve for detecting *fail* (x = false-alarm rate, y = fail recall)."""
    fpr, tpr, _ = roc_curve(1 - y, 1 - proba_pass)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    """Cross-validation outcome for one model in one feature setting.

    Attributes:
        setting: ``"A"`` or ``"B"``.
        model: Model key (see :data:`MODEL_NAMES`) or ``"baseline"``.
        y: True labels in dataset order.
        oof_proba: Out-of-fold P(pass) in dataset order.
        fold_metrics: One row of metrics per fold (threshold 0.5).
    """

    setting: str
    model: str
    y: np.ndarray
    oof_proba: np.ndarray
    fold_metrics: pd.DataFrame = field(repr=False)

    @property
    def label(self) -> str:
        """Display name of the model."""
        return MODEL_NAMES.get(self.model, BASELINE_LABEL)

    def summary(self) -> dict[str, float]:
        """Mean and std of every metric across folds."""
        out: dict[str, float] = {}
        for metric in METRIC_LABELS:
            out[metric] = float(self.fold_metrics[metric].mean())
            out[f"{metric}_std"] = float(self.fold_metrics[metric].std(ddof=1))
        return out


def cross_validate(
    df: pd.DataFrame, setting: Setting, name: str, class_weight: ClassWeight = None
) -> EvalResult:
    """Run stratified k-fold CV for one model/setting and collect fold metrics and OOF probs."""
    y = target(df)
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=config.CV_SHUFFLE, random_state=config.RANDOM_STATE
    )
    oof = np.zeros(len(df), dtype=float)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(df, y)):
        pipe = build_pipeline(df, setting, name, class_weight)
        pipe.fit(df.iloc[train_idx], y[train_idx])
        proba = _proba_pass(pipe, df.iloc[test_idx])
        oof[test_idx] = proba
        rows.append({"fold": fold + 1, **metrics_at(y[test_idx], proba)})
    return EvalResult(setting, name, y, oof, pd.DataFrame(rows))


def evaluate_all(
    df: pd.DataFrame, class_weight: ClassWeight = None
) -> dict[tuple[str, str], EvalResult]:
    """Cross-validate the baseline and all three models for settings A and B."""
    results: dict[tuple[str, str], EvalResult] = {}
    for setting in config.FEATURE_SETTINGS:
        for name in (BASELINE, *MODEL_NAMES):
            results[(setting, name)] = cross_validate(df, setting, name, class_weight)
    return results


def results_table(results: dict[tuple[str, str], EvalResult]) -> pd.DataFrame:
    """Tidy table: one row per (setting, model) with mean and std for every metric."""
    rows = [
        {"setting": s, "model": r.label, "model_key": m, **r.summary()}
        for (s, m), r in results.items()
    ]
    return pd.DataFrame(rows)


def format_results_markdown(table: pd.DataFrame) -> str:
    """Render the results table as Markdown with ``mean ± std`` cells."""
    metrics = list(METRIC_LABELS)
    header = "| Setting | Model | " + " | ".join(METRIC_LABELS[m] for m in metrics) + " |"
    lines = [header, "|" + "---|" * (len(metrics) + 2)]
    for row in table.itertuples(index=False):
        cells = [
            f"{getattr(row, m):.3f} ± {getattr(row, m + '_std'):.3f}"
            if not np.isnan(getattr(row, m))
            else "n/a"
            for m in metrics
        ]
        lines.append(f"| {row.setting} | {row.model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
def fit(
    df: pd.DataFrame, setting: Setting, name: str, class_weight: ClassWeight = None
) -> Pipeline:
    """Fit a pipeline on all rows of ``df``."""
    pipe = build_pipeline(df, setting, name, class_weight)
    pipe.fit(df, target(df))
    return pipe


def feature_importance(pipe: Pipeline) -> pd.Series:
    """Random Forest / tree impurity importances by transformed feature, descending."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    return pd.Series(pipe.named_steps["model"].feature_importances_, index=names).sort_values(
        ascending=False
    )


def standardized_coefficients(pipe: Pipeline) -> pd.Series:
    """Logistic Regression coefficients (log-odds of passing per 1 SD / category), by |size|."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pd.Series(pipe.named_steps["model"].coef_[0], index=names)
    return coefs.reindex(coefs.abs().sort_values(ascending=False).index)


def _proba_pass(pipe: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """P(pass) for each row, robust to single-class training folds."""
    proba = pipe.predict_proba(df)
    classes = list(pipe.classes_)
    if 1 not in classes:
        return np.zeros(len(df))
    return proba[:, classes.index(1)]


def predict_fail_proba(pipe: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """P(fail) for each row of ``df``."""
    return 1 - _proba_pass(pipe, df)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelBundle:
    """A fitted pipeline plus the metadata needed to reuse it safely."""

    pipeline: Pipeline
    metadata: dict[str, object]


def save_model(
    pipe: Pipeline,
    setting: Setting,
    name: str,
    class_weight: ClassWeight,
    threshold: float,
    metrics: dict[str, float],
    n_rows: int,
    models_dir: Path | None = None,
) -> Path:
    """Persist a fitted pipeline with metadata; also mark it as the latest model.

    Returns:
        Path to the ``.joblib`` file.
    """
    models_dir = models_dir or config.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = models_dir / f"model_{setting}_{name}_{stamp}.joblib"
    numeric, categorical = feature_columns_from(pipe)
    metadata = {
        "setting": setting,
        "setting_label": config.FEATURE_SETTINGS[setting],
        "model": name,
        "model_label": MODEL_NAMES.get(name, name),
        "params": {k: _jsonable(v) for k, v in pipe.named_steps["model"].get_params().items()},
        "class_weight": class_weight,
        "threshold": float(threshold),
        "metrics_cv": {k: float(v) for k, v in metrics.items()},
        "n_training_rows": int(n_rows),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "sklearn_version": sklearn.__version__,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file": path.name,
    }
    joblib.dump(ModelBundle(pipe, metadata), path)
    (models_dir / "latest.json").write_text(json.dumps(metadata, indent=2))
    return path


def load_latest(models_dir: Path | None = None) -> ModelBundle | None:
    """Load the most recently saved model, or ``None`` if there is none or it is unreadable."""
    models_dir = models_dir or config.MODELS_DIR
    pointer = models_dir / "latest.json"
    if not pointer.exists():
        return None
    try:
        meta = json.loads(pointer.read_text())
        bundle = joblib.load(models_dir / meta["file"])
    except (OSError, KeyError, ValueError, EOFError):
        return None
    return bundle if isinstance(bundle, ModelBundle) else None


def feature_columns_from(pipe: Pipeline) -> tuple[list[str], list[str]]:
    """Return the ``(numeric, categorical)`` input columns a fitted pipeline expects."""
    cols = {
        name: list(c) for name, _, c in pipe.named_steps["pre"].transformers_ if name != "remainder"
    }
    return cols.get("num", []), cols.get("cat", [])


def _jsonable(value: object) -> object:
    """Convert a hyper-parameter value into something JSON can store."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
