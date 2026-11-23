"""Generate a SYNTHETIC student dataset in the UCI Student Performance format.

The files are computer-generated for demos and testing. They are NOT the real UCI data and
must not be used to draw conclusions about real students. Column names, value codes, the
semicolon delimiter and the row counts (395 Mathematics, 649 Portuguese) match the UCI files,
and grades follow plausible patterns (past failures, study time, absences, aspiration and
parental education influence G1; G2 and G3 follow earlier grades; a few G3 = 0
non-appearances), so every chart and model in the dashboard has something to show.

Usage::

    python scripts/make_sample_data.py                   # writes data/sample/
    python scripts/make_sample_data.py --seed 7 --out some/folder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

DEFAULT_OUT = config.DATA_DIR / "sample"
FILE_NAMES = {"mat": "synthetic-student-mat.csv", "por": "synthetic-student-por.csv"}
N_POR = config.EXPECTED_ROWS["por"]
N_MAT = config.EXPECTED_ROWS["mat"]
N_SHARED = 370  # students who take both subjects (share their background attributes)


def _choice(rng: np.random.Generator, values: list, probs: list[float], n: int) -> np.ndarray:
    return rng.choice(values, size=n, p=np.asarray(probs) / np.sum(probs))


def make_students(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Background attributes that stay the same for a student across subjects."""
    school = _choice(rng, ["GP", "MS"], [0.65, 0.35], n)
    df = pd.DataFrame(
        {
            "school": school,
            "sex": _choice(rng, ["F", "M"], [0.56, 0.44], n),
            "age": _choice(
                rng, [15, 16, 17, 18, 19, 20, 21, 22], [17, 27, 27, 21, 6, 1, 0.5, 0.5], n
            ),
            "address": np.where(
                school == "GP",
                _choice(rng, ["U", "R"], [0.82, 0.18], n),
                _choice(rng, ["U", "R"], [0.45, 0.55], n),
            ),
            "famsize": _choice(rng, ["GT3", "LE3"], [0.7, 0.3], n),
            "Pstatus": _choice(rng, ["T", "A"], [0.88, 0.12], n),
            "Medu": _choice(rng, [0, 1, 2, 3, 4], [1, 20, 27, 22, 30], n),
            "Fedu": _choice(rng, [0, 1, 2, 3, 4], [1, 25, 30, 22, 22], n),
            "Mjob": _choice(
                rng, ["teacher", "health", "services", "at_home", "other"], [14, 8, 24, 19, 35], n
            ),
            "Fjob": _choice(
                rng, ["teacher", "health", "services", "at_home", "other"], [6, 4, 28, 6, 56], n
            ),
            "reason": _choice(rng, ["home", "reputation", "course", "other"], [24, 24, 42, 10], n),
            "nursery": _choice(rng, ["yes", "no"], [0.8, 0.2], n),
            "internet": _choice(rng, ["yes", "no"], [0.8, 0.2], n),
            "guardian": _choice(rng, ["mother", "father", "other"], [70, 23, 7], n),
            "traveltime": _choice(rng, [1, 2, 3, 4], [60, 30, 8, 2], n),
            "famrel": _choice(rng, [1, 2, 3, 4, 5], [2, 4, 16, 50, 28], n),
            "romantic": _choice(rng, ["yes", "no"], [0.35, 0.65], n),
            "higher": _choice(rng, ["yes", "no"], [0.92, 0.08], n),
        }
    )
    # Older students in the same grade tend to have failed before.
    df["age"] = np.where(
        df["school"] == "MS", np.minimum(df["age"] + rng.integers(0, 2, n), 22), df["age"]
    )
    return df


def add_subject_fields(
    rng: np.random.Generator, students: pd.DataFrame, subject: str
) -> pd.DataFrame:
    """Habits, support and grades for one subject, with plausible relationships."""
    n = len(students)
    df = students.copy()
    older = (df["age"] - 15).clip(lower=0).to_numpy()
    fail_rate = (
        0.12 + 0.12 * older + (0.15 if subject == "mat" else 0.05) + 0.2 * (df["higher"] == "no")
    )
    df["failures"] = np.minimum(rng.poisson(fail_rate.clip(0, 2)), 3)
    df["studytime"] = _choice(rng, [1, 2, 3, 4], [27, 49, 16, 8], n)
    df.loc[df["sex"] == "F", "studytime"] = np.minimum(
        df.loc[df["sex"] == "F", "studytime"] + rng.integers(0, 2, (df["sex"] == "F").sum()), 4
    )
    df["schoolsup"] = np.where(rng.random(n) < 0.12 + 0.05 * df["failures"], "yes", "no")
    df["famsup"] = np.where(rng.random(n) < 0.6, "yes", "no")
    df["paid"] = np.where(rng.random(n) < (0.45 if subject == "mat" else 0.06), "yes", "no")
    df["activities"] = np.where(rng.random(n) < 0.5, "yes", "no")
    df["freetime"] = _choice(rng, [1, 2, 3, 4, 5], [5, 16, 40, 29, 10], n)
    df["goout"] = np.clip(df["freetime"] + rng.integers(-1, 2, n), 1, 5)
    df["Dalc"] = _choice(rng, [1, 2, 3, 4, 5], [70, 19, 6, 3, 2], n)
    df["Walc"] = np.clip(df["Dalc"] + rng.integers(0, 3, n), 1, 5)
    df["health"] = _choice(rng, [1, 2, 3, 4, 5], [12, 12, 20, 17, 39], n)
    lam = 2.5 + 1.5 * (df["Walc"] - 1) + 2.0 * df["failures"] + (3 if subject == "mat" else 0)
    df["absences"] = np.minimum(rng.negative_binomial(1.2, 1.2 / (1.2 + lam)), 75)

    ability = rng.normal(0, 2.2, n)
    base = 11.5 if subject == "por" else 10.8
    g1 = (
        base
        + ability
        - 1.6 * df["failures"]
        + 0.6 * (df["studytime"] - 2)
        + 0.35 * (df["Medu"] - 2)
        + 1.4 * (df["higher"] == "yes")
        - 1.4
        - 0.04 * df["absences"]
        - 0.3 * (df["goout"] - 3)
        - 0.25 * (df["Dalc"] - 1)
        + 0.8 * (df["school"] == "GP")
        - 0.4
        - 0.5 * (df["schoolsup"] == "yes")
        + rng.normal(0, 1.2, n)
    )
    g2 = g1 + rng.normal(0.1, 1.2, n)
    g3 = g2 + rng.normal(0.15, 1.0, n)
    df["G1"] = np.clip(np.round(g1), 3, 19).astype(int)
    df["G2"] = np.clip(np.round(g2), 0, 19).astype(int)
    df["G3"] = np.clip(np.round(g3), 0, 20).astype(int)
    # A few students do not sit the final exam (recorded as G3 = 0).
    absent = rng.random(n) < (0.09 if subject == "mat" else 0.025) + 0.03 * df["failures"]
    df.loc[absent, "G3"] = 0
    df.loc[absent & (rng.random(n) < 0.5), "G2"] = 0
    return df[list(config.REQUIRED_COLUMNS)]


def generate(seed: int = config.RANDOM_STATE) -> dict[str, pd.DataFrame]:
    """Return ``{"mat": DataFrame, "por": DataFrame}`` in the UCI column order."""
    rng = np.random.default_rng(seed)
    students = make_students(rng, N_POR + N_MAT - N_SHARED)
    por_students = students.iloc[:N_POR].reset_index(drop=True)
    mat_students = pd.concat(
        [students.iloc[:N_SHARED], students.iloc[N_POR:]], ignore_index=True
    )  # first N_SHARED students take both subjects
    return {
        "mat": add_subject_fields(rng, mat_students, "mat"),
        "por": add_subject_fields(rng, por_students, "por"),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for subject, frame in generate(args.seed).items():
        path = args.out / FILE_NAMES[subject]
        frame.to_csv(path, sep=config.CSV_SEPARATOR, index=False)
        print(f"Wrote {path} ({len(frame)} rows, pass rate {(frame['G3'] >= 10).mean():.1%})")
    print("These files are SYNTHETIC (computer-generated), not the real UCI dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
