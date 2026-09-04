"""Persists SecurityEvent records and supports the historical-search filters
Phase 3 needs. Reuses src/events/types.SecurityEvent as the one event model —
this repository only knows how to save/load rows shaped like it, never a
second competing "Event" model."""
import datetime
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.storage.db import Database, dumps, loads


@dataclass
class StoredEvent:
    """One historical event row, enriched with the display-only fields
    (camera_name, description, snapshot_path) an event's own dataclass
    doesn't carry — never a second event *model*, just what got stored
    alongside it."""

    id: str
    camera_id: str
    camera_name: str
    source_type: Optional[str]
    event_type: str
    severity: str
    timestamp: float
    created_at: str
    track_id: Optional[int]
    identity: Optional[str]
    zone_id: Optional[str]
    zone_name: Optional[str]
    description: str
    metadata: Dict[str, Any]
    snapshot_path: Optional[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "StoredEvent":
        return cls(
            id=row["id"],
            camera_id=row["camera_id"],
            camera_name=row["camera_name"],
            source_type=row["source_type"],
            event_type=row["event_type"],
            severity=row["severity"],
            timestamp=row["timestamp"],
            created_at=row["created_at"],
            track_id=row["track_id"],
            identity=row["identity"],
            zone_id=row["zone_id"],
            zone_name=row["zone_name"],
            description=row["description"],
            metadata=loads(row["metadata"]) or {},
            snapshot_path=row["snapshot_path"],
        )


class EventRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(
        self,
        event_id: str,
        camera_id: str,
        camera_name: str,
        event_type: str,
        severity: str,
        timestamp: float,
        description: str,
        source_type: Optional[str] = None,
        track_id: Optional[int] = None,
        identity: Optional[str] = None,
        zone_id: Optional[str] = None,
        zone_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        snapshot_path: Optional[str] = None,
    ) -> StoredEvent:
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO events (
                    id, camera_id, camera_name, source_type, event_type, severity,
                    timestamp, created_at, track_id, identity, zone_id, zone_name,
                    description, metadata, snapshot_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id, camera_id, camera_name, source_type, event_type, severity,
                    timestamp, created_at, track_id, identity, zone_id, zone_name,
                    description, dumps(metadata), snapshot_path,
                ),
            )
            conn.commit()
        return StoredEvent(
            id=event_id, camera_id=camera_id, camera_name=camera_name, source_type=source_type,
            event_type=event_type, severity=severity, timestamp=timestamp, created_at=created_at,
            track_id=track_id, identity=identity, zone_id=zone_id, zone_name=zone_name,
            description=description, metadata=metadata or {}, snapshot_path=snapshot_path,
        )

    def get(self, event_id: str) -> Optional[StoredEvent]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return StoredEvent.from_row(row) if row else None

    def query(
        self,
        camera_id: Optional[str] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        identity: Optional[str] = None,
        track_id: Optional[int] = None,
        status: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[StoredEvent]:
        """`status` filters by the linked alert's lifecycle status (events
        and alerts are 1:1 via event_id) — an honest join, not a fabricated
        field on the event itself. start_time/end_time compare against
        created_at (ISO8601 wall-clock), so filters survive a restart the
        same way the video-relative `timestamp` field would not."""
        limit = max(1, min(int(limit), 500))  # never let a bad request pull unlimited rows
        offset = max(0, int(offset))

        clauses = []
        params: List[Any] = []
        joins = ""

        if camera_id:
            clauses.append("events.camera_id = ?")
            params.append(camera_id)
        if event_type:
            clauses.append("events.event_type = ?")
            params.append(event_type)
        if severity:
            clauses.append("events.severity = ?")
            params.append(severity)
        if identity:
            clauses.append("events.identity = ?")
            params.append(identity)
        if track_id is not None:
            clauses.append("events.track_id = ?")
            params.append(track_id)
        if start_time:
            clauses.append("events.created_at >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("events.created_at <= ?")
            params.append(end_time)
        if status:
            joins = "JOIN alerts ON alerts.event_id = events.id"
            clauses.append("alerts.status = ?")
            params.append(status)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT events.* FROM events {joins} {where}
            ORDER BY events.created_at DESC
            LIMIT ? OFFSET ?
        """
        params += [limit, offset]

        with self.db.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [StoredEvent.from_row(r) for r in rows]

    def related_to(self, camera_id: str, track_id: int, exclude_event_id: Optional[str] = None, limit: int = 20) -> List[StoredEvent]:
        """Other events for the same (camera, track) — the only relationship
        the pipeline actually knows, never an inferred cross-camera link."""
        sql = "SELECT * FROM events WHERE camera_id = ? AND track_id = ?"
        params: List[Any] = [camera_id, track_id]
        if exclude_event_id:
            sql += " AND id != ?"
            params.append(exclude_event_id)
        sql += " ORDER BY created_at ASC LIMIT ?"
        params.append(limit)
        with self.db.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [StoredEvent.from_row(r) for r in rows]

    def for_identity(self, identity: str, limit: int = 50) -> List[StoredEvent]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE identity = ? ORDER BY created_at DESC LIMIT ?",
                (identity, limit),
            ).fetchall()
        return [StoredEvent.from_row(r) for r in rows]
