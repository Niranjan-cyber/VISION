"""VISION Phase 3 persistence — a single local SQLite database for events,
alerts, and zones. Deliberately not PostgreSQL/Redis/Kafka: this is an SIH
prototype, and a single-file embedded database is the simplest thing that
actually satisfies "events survive a restart" and "operators can search
history." The existing PostgreSQL-backed face gallery (src/face/vector_db.py)
is a separate, optional, unrelated concern and is untouched by this module.
"""

from src.storage.db import Database, get_database
from src.storage.event_repository import EventRepository, StoredEvent
from src.storage.alert_repository import AlertRepository, InvalidAlertTransition, StoredAlert
from src.storage.zone_repository import ZoneRepository, StoredZone
from src.storage.persistence_service import EventPersistenceService

__all__ = [
    "Database",
    "get_database",
    "EventRepository",
    "StoredEvent",
    "AlertRepository",
    "InvalidAlertTransition",
    "StoredAlert",
    "ZoneRepository",
    "StoredZone",
    "EventPersistenceService",
]
