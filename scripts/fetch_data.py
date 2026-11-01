"""Download the UCI Student Performance dataset into ``data/raw/`` and print a quality summary.

Usage::

    python scripts/fetch_data.py            # download (if needed), validate, process
    python scripts/fetch_data.py --force    # re-download even if files exist
    python scripts/fetch_data.py --offline  # skip download; process existing raw files

On download failure the script prints exact manual-download instructions and exits 1.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, ingestion  # noqa: E402

TARGET_FILES: tuple[Path, ...] = (config.RAW_MAT_FILE, config.RAW_POR_FILE)


def download(url: str = config.DATASET_ZIP_URL, timeout: int = config.DOWNLOAD_TIMEOUT_S) -> bytes:
    """Download ``url`` and return the response body.

    Raises:
        urllib.error.URLError: On network failure or a non-2xx response.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "education-analytics/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def extract_targets(archive: bytes, names: tuple[str, ...]) -> dict[str, bytes]:
    """Find ``names`` inside a (possibly nested) zip archive and return their bytes.

    Only the requested members are read, by basename; nothing is written from the
    archive's own paths, so a malicious member path cannot escape ``data/raw``.

    Args:
        archive: Raw zip bytes.
        names: Basenames to find, e.g. ``("student-mat.csv", "student-por.csv")``.

    Returns:
        Mapping of basename to file bytes for every name found.
    """
    found: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name
            if base in names and base not in found:
                found[base] = zf.read(info)
            elif base.lower().endswith(".zip"):
                for k, v in extract_targets(zf.read(info), names).items():
                    found.setdefault(k, v)
    return found


def manual_instructions() -> str:
    """Return step-by-step manual-download instructions."""
    raw = config.RAW_DIR.relative_to(config.PROJECT_ROOT)
    return (
        "\nAutomatic download failed. Download the dataset manually:\n"
        f"  1. Open {config.DATASET_PAGE_URL}\n"
        f"     (direct link: {config.DATASET_ZIP_URL})\n"
        "  2. Click 'Download' and unzip 'student+performance.zip'.\n"
        "  3. Inside it, unzip 'student.zip'.\n"
        "  4. Copy these two files into the project folder:\n"
        f"       {raw / config.RAW_MAT_FILE.name}\n"
        f"       {raw / config.RAW_POR_FILE.name}\n"
        f"     (absolute target folder: {config.RAW_DIR})\n"
        "  5. Re-run: python scripts/fetch_data.py --offline\n"
    )


def fetch(force: bool = False) -> bool:
    """Ensure both raw files exist in ``data/raw``; return True on success."""
    if not force and all(p.exists() for p in TARGET_FILES):
        print(f"Raw files already present in {config.RAW_DIR} (use --force to re-download).")
        return True
    print(f"Downloading {config.DATASET_ZIP_URL} ...")
    try:
        archive = download()
        files = extract_targets(archive, tuple(p.name for p in TARGET_FILES))
    except (urllib.error.URLError, TimeoutError, OSError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(manual_instructions(), file=sys.stderr)
        return False
    missing = [p.name for p in TARGET_FILES if p.name not in files]
    if missing:
        print(f"ERROR: archive did not contain {missing}.", file=sys.stderr)
        print(manual_instructions(), file=sys.stderr)
        return False
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for path in TARGET_FILES:
        path.write_bytes(files[path.name])
        digest = ingestion.file_checksum(path)[:12]
        print(f"  saved {path.relative_to(config.PROJECT_ROOT)}  sha256={digest}…")
    return True


def process() -> int:
    """Validate and prepare the raw files, save the processed CSV and print the summary."""
    try:
        dataset, reports = ingestion.load_bundled()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(manual_instructions(), file=sys.stderr)
        return 1
    for subject, expected in config.EXPECTED_ROWS.items():
        actual = reports[subject].n_rows
        if actual != expected:
            print(f"WARNING: {subject} has {actual} rows; the published dataset has {expected}.")
    if dataset.empty:
        print("ERROR: no usable data; every file failed validation:", file=sys.stderr)
        for rep in reports.values():
            print(rep.to_frame().to_string(index=False))
        return 1
    out = ingestion.save_processed(dataset)
    print(ingestion.format_quality_summary(ingestion.data_quality_summary(dataset), reports))
    print(f"Processed dataset written to {out.relative_to(config.PROJECT_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--offline", action="store_true", help="skip download")
    args = parser.parse_args(argv)
    config.ensure_dirs()
    if not args.offline and not fetch(force=args.force):
        return 1
    return process()


if __name__ == "__main__":
    raise SystemExit(main())
