"""Persists surveillance zones so they survive a backend restart and can be
managed from the UI. YAML zone files (configs/zones_*.yaml) remain the
golden-demo bootstrap source — see src/pipeline/session.py's zone-loading
helper, which seeds this table from YAML once per camera_id and treats this
table as the source of truth from then on."""
import datetime
import sqlite3
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from src.storage.db import Database, dumps, loads

VALID_ZONE_TYPES = {"restricted", "warning", "monitored"}


@dataclass
class StoredZone:
    id: str
    camera_id: str
    name: str
    type: str
    polygon: List[Tuple[int, int]]
    enabled: bool
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "StoredZone":
        return cls(
            id=row["id"], camera_id=row["camera_id"], name=row["name"], type=row["type"],
            polygon=[tuple(p) for p in loads(row["polygon"])],
            enabled=bool(row["enabled"]), created_at=row["created_at"], updated_at=row["updated_at"],
        )


def validate_polygon(polygon: Any) -> List[Tuple[int, int]]:
    if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
        raise ValueError("polygon must be a list of at least 3 [x, y] points")
    validated: List[Tuple[int, int]] = []
    for i, pt in enumerate(polygon):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ValueError(f"polygon vertex #{i} must be a 2-element [x, y] point, got {pt}")
        try:
            x, y = int(pt[0]), int(pt[1])
        except (TypeError, ValueError):
            raise ValueError(f"polygon vertex #{i} coordinates must be numeric, got {pt}")
        if x < 0 or y < 0:
            raise ValueError(f"polygon vertex #{i} coordinates must be non-negative, got ({x}, {y})")
        validated.append((x, y))

    xs = [p[0] for p in validated]
    ys = [p[1] for p in validated]
    if max(xs) == min(xs) or max(ys) == min(ys):
        raise ValueError("polygon has zero area — all vertices are collinear or identical")

    return validated


class ZoneRepository:
    def __init__(self, db: Database):
        self.db = db

    def list_for_camera(self, camera_id: str) -> List[StoredZone]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM zones WHERE camera_id = ? ORDER BY created_at ASC", (camera_id,)
            ).fetchall()
        return [StoredZone.from_row(r) for r in rows]

    def get(self, zone_id: str) -> Optional[StoredZone]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM zones WHERE id = ?", (zone_id,)).fetchone()
        return StoredZone.from_row(row) if row else None

    def create(self, zone_id: str, camera_id: str, name: str, zone_type: str, polygon, enabled: bool = True) -> StoredZone:
        if zone_type not in VALID_ZONE_TYPES:
            raise ValueError(f"type must be one of {sorted(VALID_ZONE_TYPES)}, got '{zone_type}'")
        if not name or not name.strip():
            raise ValueError("name must be a non-empty string")
        validated_polygon = validate_polygon(polygon)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO zones (id, camera_id, name, type, polygon, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (zone_id, camera_id, name.strip(), zone_type, dumps(validated_polygon), int(enabled), now, now),
            )
            conn.commit()
        return StoredZone(
            id=zone_id, camera_id=camera_id, name=name.strip(), type=zone_type,
            polygon=validated_polygon, enabled=enabled, created_at=now, updated_at=now,
        )

    def update(
        self,
        zone_id: str,
        name: Optional[str] = None,
        zone_type: Optional[str] = None,
        polygon: Optional[Any] = None,
        enabled: Optional[bool] = None,
    ) -> StoredZone:
        existing = self.get(zone_id)
        if existing is None:
            raise ValueError(f"No zone with id '{zone_id}'")

        new_name = existing.name
        if name is not None:
            if not name.strip():
                raise ValueError("name must be a non-empty string")
            new_name = name.strip()

        new_type = existing.type
        if zone_type is not None:
            if zone_type not in VALID_ZONE_TYPES:
                raise ValueError(f"type must be one of {sorted(VALID_ZONE_TYPES)}, got '{zone_type}'")
            new_type = zone_type

        new_polygon = existing.polygon
        if polygon is not None:
            new_polygon = validate_polygon(polygon)

        new_enabled = existing.enabled if enabled is None else bool(enabled)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with self.db.connect() as conn:
            conn.execute(
                "UPDATE zones SET name = ?, type = ?, polygon = ?, enabled = ?, updated_at = ? WHERE id = ?",
                (new_name, new_type, dumps(new_polygon), int(new_enabled), now, zone_id),
            )
            conn.commit()
        return StoredZone(
            id=zone_id, camera_id=existing.camera_id, name=new_name, type=new_type,
            polygon=new_polygon, enabled=new_enabled, created_at=existing.created_at, updated_at=now,
        )

    def delete(self, zone_id: str) -> bool:
        with self.db.connect() as conn:
            cur = conn.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
            conn.commit()
            return cur.rowcount > 0

    def seed_if_empty(self, camera_id: str, zones: List[Any]) -> None:
        """Seeds this camera's zones from an already-loaded YAML zone list,
        but only the very first time (an empty table for this camera_id) —
        never overwrites an operator's edits on a later restart."""
        if self.list_for_camera(camera_id):
            return
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.db.connect() as conn:
            for z in zones:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO zones (id, camera_id, name, type, polygon, enabled, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (z.id, camera_id, z.name, z.zone_type, dumps([list(p) for p in z.polygon]), now, now),
                )
            conn.commit()
