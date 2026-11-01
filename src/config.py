"""Central configuration: paths, constants, colour theme and column schema.

Every tunable value used anywhere in the project lives here so that the UI pages,
scripts and library modules stay free of magic numbers and hard-coded paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths (all relative to the project root; pathlib only, no absolute paths)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
METRICS_DIR: Final[Path] = REPORTS_DIR / "metrics"
DB_PATH: Final[Path] = DATA_DIR / "app.db"

RAW_MAT_FILE: Final[Path] = RAW_DIR / "student-mat.csv"
RAW_POR_FILE: Final[Path] = RAW_DIR / "student-por.csv"
PROCESSED_FILE: Final[Path] = PROCESSED_DIR / "students_clean.csv"

ALL_DIRS: Final[tuple[Path, ...]] = (
    RAW_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
)

# ---------------------------------------------------------------------------
# Dataset source
# ---------------------------------------------------------------------------
DATASET_PAGE_URL: Final[str] = "https://archive.ics.uci.edu/dataset/320/student+performance"
DATASET_ZIP_URL: Final[str] = (
    "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
)
DATASET_CITATION: Final[str] = (
    "P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student "
    "Performance. In A. Brito and J. Teixeira (Eds.), Proceedings of 5th FUture BUsiness "
    "TEChnology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, April 2008, EUROSIS."
)
CSV_SEPARATOR: Final[str] = ";"
EXPECTED_ROWS: Final[dict[str, int]] = {"mat": 395, "por": 649}
SUBJECT_LABELS: Final[dict[str, str]] = {"mat": "Mathematics", "por": "Portuguese"}

# ---------------------------------------------------------------------------
# Column schema (33 original attributes)
# ---------------------------------------------------------------------------
GRADE_MIN: Final[int] = 0
GRADE_MAX: Final[int] = 20
GRADE_COLUMNS: Final[tuple[str, ...]] = ("G1", "G2", "G3")
TARGET_SOURCE: Final[str] = "G3"


@dataclass(frozen=True)
class ColumnSpec:
    """Schema rule for a single input column.

    Attributes:
        kind: ``"numeric"`` or ``"categorical"``.
        min_value: Inclusive lower bound for numeric columns.
        max_value: Inclusive upper bound for numeric columns.
        allowed: Permitted values for categorical columns.
        description: Human-readable meaning (from the UCI codebook).
    """

    kind: str
    description: str
    min_value: int | None = None
    max_value: int | None = None
    allowed: tuple[str, ...] | None = None


_YES_NO: Final[tuple[str, ...]] = ("yes", "no")
_JOBS: Final[tuple[str, ...]] = ("teacher", "health", "services", "at_home", "other")

SCHEMA: Final[dict[str, ColumnSpec]] = {
    "school": ColumnSpec("categorical", "Student's school", allowed=("GP", "MS")),
    "sex": ColumnSpec("categorical", "Student's sex", allowed=("F", "M")),
    "age": ColumnSpec("numeric", "Student's age (years)", 15, 22),
    "address": ColumnSpec("categorical", "Home address type (Urban/Rural)", allowed=("U", "R")),
    "famsize": ColumnSpec("categorical", "Family size", allowed=("LE3", "GT3")),
    "Pstatus": ColumnSpec("categorical", "Parents' cohabitation status", allowed=("T", "A")),
    "Medu": ColumnSpec("numeric", "Mother's education (0 none – 4 higher)", 0, 4),
    "Fedu": ColumnSpec("numeric", "Father's education (0 none – 4 higher)", 0, 4),
    "Mjob": ColumnSpec("categorical", "Mother's job", allowed=_JOBS),
    "Fjob": ColumnSpec("categorical", "Father's job", allowed=_JOBS),
    "reason": ColumnSpec(
        "categorical",
        "Reason to choose this school",
        allowed=("home", "reputation", "course", "other"),
    ),
    "guardian": ColumnSpec(
        "categorical", "Student's guardian", allowed=("mother", "father", "other")
    ),
    "traveltime": ColumnSpec("numeric", "Home-to-school travel time (1 <15m – 4 >1h)", 1, 4),
    "studytime": ColumnSpec("numeric", "Weekly study time (1 <2h – 4 >10h)", 1, 4),
    "failures": ColumnSpec("numeric", "Number of past class failures", 0, 4),
    "schoolsup": ColumnSpec("categorical", "Extra educational support", allowed=_YES_NO),
    "famsup": ColumnSpec("categorical", "Family educational support", allowed=_YES_NO),
    "paid": ColumnSpec("categorical", "Extra paid classes", allowed=_YES_NO),
    "activities": ColumnSpec("categorical", "Extra-curricular activities", allowed=_YES_NO),
    "nursery": ColumnSpec("categorical", "Attended nursery school", allowed=_YES_NO),
    "higher": ColumnSpec("categorical", "Wants higher education", allowed=_YES_NO),
    "internet": ColumnSpec("categorical", "Internet access at home", allowed=_YES_NO),
    "romantic": ColumnSpec("categorical", "In a romantic relationship", allowed=_YES_NO),
    "famrel": ColumnSpec("numeric", "Quality of family relationships (1–5)", 1, 5),
    "freetime": ColumnSpec("numeric", "Free time after school (1–5)", 1, 5),
    "goout": ColumnSpec("numeric", "Going out with friends (1–5)", 1, 5),
    "Dalc": ColumnSpec("numeric", "Workday alcohol consumption (1–5)", 1, 5),
    "Walc": ColumnSpec("numeric", "Weekend alcohol consumption (1–5)", 1, 5),
    "health": ColumnSpec("numeric", "Current health status (1–5)", 1, 5),
    "absences": ColumnSpec("numeric", "Number of school absences", 0, 93),
    "G1": ColumnSpec("numeric", "First period grade", GRADE_MIN, GRADE_MAX),
    "G2": ColumnSpec("numeric", "Second period grade", GRADE_MIN, GRADE_MAX),
    "G3": ColumnSpec("numeric", "Final grade (target source)", GRADE_MIN, GRADE_MAX),
}
REQUIRED_COLUMNS: Final[tuple[str, ...]] = tuple(SCHEMA)
NUMERIC_COLUMNS: Final[tuple[str, ...]] = tuple(c for c, s in SCHEMA.items() if s.kind == "numeric")
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = tuple(
    c for c, s in SCHEMA.items() if s.kind == "categorical"
)

# Attributes the original authors used to match the same student across both files.
STUDENT_MATCH_KEYS: Final[tuple[str, ...]] = (
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "nursery", "internet",
)  # fmt: skip

# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------
SUBJECT_COL: Final[str] = "subject"
PASS_COL: Final[str] = "pass"
ABSENCE_BAND_COL: Final[str] = "absence_band"
PASS_THRESHOLD: Final[int] = 10  # pass = 1 if G3 >= 10

# Bands: 0 | 1–5 | 6–10 | 11–20 | 20+   (right-inclusive edges)
ABSENCE_BAND_EDGES: Final[tuple[float, ...]] = (-0.5, 0.5, 5.5, 10.5, 20.5, float("inf"))
ABSENCE_BAND_LABELS: Final[tuple[str, ...]] = ("0", "1–5", "6–10", "11–20", "20+")

PASS_LABEL: Final[str] = "Pass"
FAIL_LABEL: Final[str] = "Fail"

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
RANDOM_STATE: Final[int] = 42
CV_FOLDS: Final[int] = 5
CV_SHUFFLE: Final[bool] = True
DT_MAX_DEPTH: Final[int] = 5
RF_N_ESTIMATORS: Final[int] = 300
LOGREG_MAX_ITER: Final[int] = 2000
DEFAULT_THRESHOLD: Final[float] = 0.5

# Settings: A includes G1/G2; B excludes them (early-warning scenario).
FEATURE_SETTINGS: Final[dict[str, str]] = {
    "A": "With prior grades (G1, G2)",
    "B": "Without prior grades (early warning)",
}
SETTING_B_EXCLUDED: Final[tuple[str, ...]] = ("G1", "G2")
# Never used as model inputs: the target, its source and derived labels.
NON_FEATURE_COLUMNS: Final[tuple[str, ...]] = ("G3", PASS_COL, ABSENCE_BAND_COL)

# Early-warning risk tiers on P(fail).
RISK_HIGH: Final[float] = 0.7
RISK_MEDIUM: Final[float] = 0.4
TOP_N_REASONS: Final[int] = 3

# ---------------------------------------------------------------------------
# Colour theme — Okabe-Ito colour-blind-safe palette
# ---------------------------------------------------------------------------
OKABE_ITO: Final[dict[str, str]] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
PASS_COLOR: Final[str] = OKABE_ITO["blue"]
FAIL_COLOR: Final[str] = OKABE_ITO["vermillion"]
OUTCOME_COLORS: Final[dict[str, str]] = {PASS_LABEL: PASS_COLOR, FAIL_LABEL: FAIL_COLOR}
# Pass/fail must never rely on colour alone: pair colour with a pattern + text label.
OUTCOME_PATTERNS: Final[dict[str, str]] = {PASS_LABEL: "", FAIL_LABEL: "/"}
CATEGORICAL_SEQUENCE: Final[tuple[str, ...]] = (
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["green"],
    OKABE_ITO["purple"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["yellow"],
)
DIVERGING_SCALE: Final[str] = "RdBu"  # colour-blind-safe diverging scale for heatmaps
RISK_COLORS: Final[dict[str, str]] = {
    "High": OKABE_ITO["vermillion"],
    "Medium": OKABE_ITO["orange"],
    "Low": OKABE_ITO["sky_blue"],
}
FONT_FAMILY: Final[str] = "Inter, Segoe UI, Helvetica, Arial, sans-serif"
BASE_FONT_SIZE: Final[int] = 14
FIGURE_EXPORT_DPI: Final[int] = 300

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
APP_TITLE: Final[str] = "Education Analytics Dashboard"
APP_ICON: Final[str] = "🎓"
UPLOAD_TYPES: Final[tuple[str, ...]] = ("csv", "xlsx")


def ensure_dirs() -> None:
    """Create every project output directory if it does not already exist."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
