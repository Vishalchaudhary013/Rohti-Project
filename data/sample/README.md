# Synthetic sample dataset

`synthetic-student-mat.csv` (395 rows) and `synthetic-student-por.csv` (649 rows) are
**computer-generated** by `scripts/make_sample_data.py` (seed 42). They are **not** the real
UCI Student Performance data and describe no real students. Use them to try the dashboard;
do not draw conclusions about real students from them.

- Same 33 columns, value codes and `;` delimiter as the UCI files, so they upload directly.
- Plausible patterns: past failures, study time, absences, parental education and
  higher-education aspiration influence grades; G2 and G3 follow earlier grades; a few G3 = 0
  rows imitate students who missed the final exam; 370 students appear in both subjects.

Regenerate with `python scripts/make_sample_data.py`. For real analysis, use the UCI files:
<https://archive.ics.uci.edu/dataset/320/student+performance>.
