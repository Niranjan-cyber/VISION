"""Persists operational Alert records (src/events/types.Alert) and enforces
the NEW -> ACKNOWLEDGED -> RESOLVED lifecycle. This is the one place a
status transition is validated — the API layer and any other caller goes
through here rather than mutating status directly."""
import datetime
import sqlite3
from dataclasses import dataclass
from typing import Any, List, Optional

from src.storage.db import Database

VALID_TRANSITIONS = {
    "NEW": {"ACKNOWLEDGED", "RESOLVED"},
    "ACKNOWLEDGED": {"RESOLVED"},
    "RESOLVED": set(),
}


class InvalidAlertTransition(ValueError):
    def __init__(self, current: str, target: str):
        self.current = current
        self.target = target
        super().__init__(f"Cannot transition alert from {current} to {target}")


@dataclass
class StoredAlert:
    id: str
    event_id: str
    camera_id: str
    camera_name: str
    severity: str
    title: str
    message: str
    status: str
    created_at: str
    acknowledged_at: Optional[str]
    resolved_at: Optional[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "StoredAlert":
        return cls(
            id=row["id"], event_id=row["event_id"], camera_id=row["camera_id"],
            camera_name=row["camera_name"], severity=row["severity"], title=row["title"],
            message=row["message"], status=row["status"], created_at=row["created_at"],
            acknowledged_at=row["acknowledged_at"], resolved_at=row["resolved_at"],
        )


class AlertRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(
        self,
        alert_id: str,
        event_id: str,
        camera_id: str,
        camera_name: str,
        severity: str,
        title: str,
        message: str,
    ) -> StoredAlert:
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO alerts (id, event_id, camera_id, camera_name, severity, title, message, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'NEW', ?)
                """,
                (alert_id, event_id, camera_id, camera_name, severity, title, message, created_at),
            )
            conn.commit()
        return StoredAlert(
            id=alert_id, event_id=event_id, camera_id=camera_id, camera_name=camera_name,
            severity=severity, title=title, message=message, status="NEW",
            created_at=created_at, acknowledged_at=None, resolved_at=None,
        )

    def get(self, alert_id: str) -> Optional[StoredAlert]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return StoredAlert.from_row(row) if row else None

    def get_by_event(self, event_id: str) -> Optional[StoredAlert]:
        """Events and alerts are 1:1 (see src/events/engine.py's
        _create_event_and_alert) — this is the lookup investigation views
        use to show an event's current alert status."""
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM alerts WHERE event_id = ?", (event_id,)).fetchone()
        return StoredAlert.from_row(row) if row else None

    def query(
        self,
        camera_id: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[StoredAlert]:
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))

        clauses = []
        params: List[Any] = []
        joins = ""

        if camera_id:
            clauses.append("alerts.camera_id = ?")
            params.append(camera_id)
        if severity:
            clauses.append("alerts.severity = ?")
            params.append(severity)
        if status:
            clauses.append("alerts.status = ?")
            params.append(status)
        if start_time:
            clauses.append("alerts.created_at >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("alerts.created_at <= ?")
            params.append(end_time)
        if event_type:
            joins = "JOIN events ON events.id = alerts.event_id"
            clauses.append("events.event_type = ?")
            params.append(event_type)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT alerts.* FROM alerts {joins} {where}
            ORDER BY alerts.created_at DESC
            LIMIT ? OFFSET ?
        """
        params += [limit, offset]

        with self.db.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [StoredAlert.from_row(r) for r in rows]

    def transition(self, alert_id: str, target_status: str) -> StoredAlert:
        """Raises InvalidAlertTransition for a nonsensical move (e.g.
        RESOLVED -> ACKNOWLEDGED) and KeyError-free 404 signaling via
        returning None is left to the caller (matches the rest of this
        codebase's `get()` convention) — this raises ValueError if the
        alert doesn't exist at all, since that's a caller bug, not a user
        state."""
        current = self.get(alert_id)
        if current is None:
            raise ValueError(f"No alert with id '{alert_id}'")
        if target_status not in VALID_TRANSITIONS.get(current.status, set()):
            raise InvalidAlertTransition(current.status, target_status)

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        column = "acknowledged_at" if target_status == "ACKNOWLEDGED" else "resolved_at"
        with self.db.connect() as conn:
            conn.execute(
                f"UPDATE alerts SET status = ?, {column} = ? WHERE id = ?",
                (target_status, now, alert_id),
            )
            conn.commit()
        return self.get(alert_id)
