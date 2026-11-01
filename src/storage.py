"""SQLite repository layer (stdlib ``sqlite3``) for upload and prediction-run history."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    uploaded_at TEXT    NOT NULL,
    filename    TEXT    NOT NULL,
    subject     TEXT,
    row_count   INTEGER NOT NULL,
    rows_kept   INTEGER NOT NULL,
    checksum    TEXT    NOT NULL,
    status      TEXT    NOT NULL
);
"""


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a connection, ensure the schema exists, commit on success and always close."""
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def register_upload(
    filename: str,
    row_count: int,
    rows_kept: int,
    checksum: str,
    status: str,
    subject: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Record one uploaded file and return its row id.

    Args:
        filename: Original file name.
        row_count: Rows in the raw file.
        rows_kept: Rows remaining after validation and cleaning.
        checksum: SHA-256 of the file bytes.
        status: ``"accepted"`` or ``"rejected"``.
        subject: Subject code the file was loaded as, if known.
        db_path: Override the database location (tests).
    """
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO uploads (uploaded_at, filename, subject, row_count, rows_kept, "
            "checksum, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_now(), filename, subject, row_count, rows_kept, checksum, status),
        )
        return int(cur.lastrowid or 0)


def list_uploads(limit: int = 50, db_path: Path | None = None) -> pd.DataFrame:
    """Return the most recent uploads, newest first."""
    with connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT uploaded_at, filename, subject, row_count, rows_kept, status, checksum "
            "FROM uploads ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
