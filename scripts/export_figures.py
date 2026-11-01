"""Export every exploratory chart as a 300-dpi PNG to ``reports/figures/``.

Usage::

    python scripts/export_figures.py                       # bundled data/raw files
    python scripts/export_figures.py --input a.csv b.csv   # your own files
    python scripts/export_figures.py --exclude-g3-zero --dpi 150 --out some/folder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, ingestion, visuals  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", nargs="+", type=Path, help="CSV/XLSX files to use")
    parser.add_argument("--exclude-g3-zero", action="store_true", help="drop G3 == 0 rows")
    parser.add_argument("--dpi", type=int, default=config.FIGURE_EXPORT_DPI)
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR)
    args = parser.parse_args(argv)

    try:
        df = ingestion.load_files(args.input).dataset
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if df.empty:
        print("ERROR: no file passed validation.", file=sys.stderr)
        return 1
    if args.exclude_g3_zero:
        df = df[~df[config.G3_ZERO_FLAG_COL]]

    paths = visuals.export_static_figures(df, args.out, args.dpi)
    print(f"Exported {len(paths)} figures ({args.dpi} dpi, {len(df):,} rows) to {args.out}:")
    for path in paths:
        print(f"  {path.name:<36} {visuals.FIGURE_CATALOG[path.name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
