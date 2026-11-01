"""Cross-validate all models for settings A and B and write a metrics report.

Usage::

    python scripts/train.py                              # uses data/raw/student-{mat,por}.csv
    python scripts/train.py --input a.csv b.xlsx         # your own files
    python scripts/train.py --balanced --exclude-g3-zero
    python scripts/train.py --save random_forest --setting B --threshold 0.5

Writes ``reports/metrics/metrics.json`` and ``reports/metrics/metrics.md`` and prints the
Markdown table. With ``--save`` the chosen model is fitted on all rows and saved to ``models/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, ingestion, modeling  # noqa: E402


def load(inputs: list[Path] | None) -> ingestion.UploadResult:
    """Load the bundled raw files, or the given input files, into a prepared dataset."""
    paths = inputs or [config.RAW_MAT_FILE, config.RAW_POR_FILE]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing input file(s): {missing}. Pass --input, or run scripts/fetch_data.py."
        )
    items = [
        ingestion.UploadItem(p.name, ingestion.read_table(p), ingestion.infer_subject(p.name))
        for p in paths
    ]
    return ingestion.prepare_uploads(items)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", nargs="+", type=Path, help="CSV/XLSX files to use")
    parser.add_argument("--balanced", action="store_true", help='class_weight="balanced"')
    parser.add_argument("--exclude-g3-zero", action="store_true", help="drop G3 == 0 rows")
    parser.add_argument("--save", choices=list(modeling.MODEL_NAMES), help="model to persist")
    parser.add_argument("--setting", choices=list(config.FEATURE_SETTINGS), default="B")
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    parser.add_argument("--out", type=Path, default=config.METRICS_DIR, help="report folder")
    args = parser.parse_args(argv)

    try:
        loaded = load(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    df = loaded.dataset
    if args.exclude_g3_zero and not df.empty:
        df = df[~df[config.G3_ZERO_FLAG_COL]]
    ok, reason = modeling.can_model(df)
    if not ok:
        print(f"ERROR: {reason}", file=sys.stderr)
        return 1

    class_weight = "balanced" if args.balanced else None
    print(f"Cross-validating on {len(df):,} rows (class_weight={class_weight})…")
    results = modeling.evaluate_all(df, class_weight)
    table = modeling.results_table(results)
    markdown = modeling.format_results_markdown(table)

    args.out.mkdir(parents=True, exist_ok=True)
    run = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": len(df),
        "pass_rate": float(df[config.PASS_COL].mean()),
        "class_weight": class_weight,
        "exclude_g3_zero": args.exclude_g3_zero,
        "cv": {
            "folds": config.CV_FOLDS,
            "shuffle": config.CV_SHUFFLE,
            "random_state": config.RANDOM_STATE,
        },
        "results": table.drop(columns=["model_key"]).to_dict(orient="records"),
    }
    (args.out / "metrics.json").write_text(json.dumps(run, indent=2))
    (args.out / "metrics.md").write_text(
        f"# Cross-validated metrics\n\nRows: {len(df):,} · pass rate {run['pass_rate']:.1%} · "
        f"class_weight={class_weight} · {config.CV_FOLDS}-fold stratified CV, seed "
        f"{config.RANDOM_STATE}. Values are mean ± std across folds at threshold 0.5.\n\n"
        + markdown
        + "\n"
    )
    print(markdown)
    print(f"\nReport written to {args.out / 'metrics.md'} and metrics.json")

    if args.save:
        pipe = modeling.fit(df, args.setting, args.save, class_weight)
        summary = results[(args.setting, args.save)].summary()
        path = modeling.save_model(
            pipe, args.setting, args.save, class_weight, args.threshold, summary, len(df)
        )
        print(f"Saved model to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
