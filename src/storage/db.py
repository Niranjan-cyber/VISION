"""
SQLite schema + connection management for VISION's Phase 3 event/alert/zone
store. One connection per operation (never shared across threads) — up to
four CameraSession AI-worker threads and the FastAPI process all touch this
database, and sqlite3 connections are not safe to share across threads.
Write volume is low (new events/alerts only, not per-frame), so opening a
short-lived connection per call carries no meaningful cost, and WAL mode
lets concurrent readers (the API) proceed without blocking on a writer
(a camera's AI thread persisting a new event).
"""
import json
import os
import sqlite3
import sys
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

DEFAULT_DB_PATH = os.environ.get("VISION_DB_PATH", "data/vision.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    camera_name TEXT NOT NULL,
    source_type TEXT,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    timestamp REAL NOT NULL,
    created_at TEXT NOT NULL,
    track_id INTEGER,
    identity TEXT,
    zone_id TEXT,
    zone_name TEXT,
    description TEXT NOT NULL,
    metadata TEXT,
    snapshot_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity);
CREATE INDEX IF NOT EXISTS idx_events_identity ON events(identity);
CREATE INDEX IF NOT EXISTS idx_events_track ON events(camera_id, track_id);

CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(id),
    camera_id TEXT NOT NULL,
    camera_name TEXT NOT NULL,
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'NEW',
    created_at TEXT NOT NULL,
    acknowledged_at TEXT,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_camera ON alerts(camera_id);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);

CREATE TABLE IF NOT EXISTS zones (
    id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    polygon TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_zones_camera ON zones(camera_id);
"""


class Database:
    """Owns one SQLite file's schema. Thread-safe by construction: every
    call opens and closes its own connection rather than sharing one."""

    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        try:
            with self.connect() as conn:
                conn.executescript(_SCHEMA)
                conn.commit()
        except Exception as e:
            print(f"[ERROR] Failed to initialize VISION event/alert/zone database at '{self.path}': {e}", file=sys.stderr)
            raise

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()


def dumps(value) -> Optional[str]:
    """JSON-encodes metadata/geometry for storage; None stays None rather
    than becoming the literal string 'null'."""
    if value is None:
        return None
    return json.dumps(value)


def loads(value: Optional[str]):
    if value is None:
        return None
    return json.loads(value)


_lock = threading.Lock()
_default_instance: Optional[Database] = None


def get_database(path: Optional[str] = None) -> Database:
    """Process-wide default Database instance (one file, one schema init).
    Tests and callers that need an isolated database pass their own `path`
    and get a fresh, independent instance instead."""
    global _default_instance
    if path is not None:
        return Database(path)
    with _lock:
        if _default_instance is None:
            _default_instance = Database(DEFAULT_DB_PATH)
        return _default_instance
