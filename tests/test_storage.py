"""
Unit tests for the Phase 3 SQLite-backed event/alert/zone repositories.
No AI models are loaded here — these are fast, pure-storage tests, isolated
from the pipeline. Each test gets its own temp database file (never
":memory:" — Database opens a fresh connection per call by design, and an
in-memory SQLite database does not survive across connections).
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.events.types import Alert, AlertStatus, EventType, SecurityEvent, Severity
from src.storage.db import Database
from src.storage.event_repository import EventRepository
from src.storage.alert_repository import AlertRepository, InvalidAlertTransition
from src.storage.zone_repository import ZoneRepository
from src.storage.persistence_service import EventPersistenceService


class StorageTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp(prefix="vision_test_db_")
        self.db = Database(os.path.join(self._tmp_dir, "test.db"))
        self.events = EventRepository(self.db)
        self.alerts = AlertRepository(self.db)
        self.zones = ZoneRepository(self.db)

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)


class TestDatabaseSchema(StorageTestCase):
    def test_schema_tables_exist(self):
        with self.db.connect() as conn:
            names = {
                r[0]
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
        self.assertIn("events", names)
        self.assertIn("alerts", names)
        self.assertIn("zones", names)

    def test_reopening_database_at_same_path_preserves_data(self):
        self.events.save(
            event_id="evt_1", camera_id="CAM-01", camera_name="Border Gate",
            event_type="INTRUSION", severity="HIGH", timestamp=1.0, description="test",
        )
        reopened = Database(self.db.path)
        reopened_repo = EventRepository(reopened)
        self.assertIsNotNone(reopened_repo.get("evt_1"))


class TestEventRepository(StorageTestCase):
    def test_save_and_get_event(self):
        saved = self.events.save(
            event_id="evt_1", camera_id="CAM-01", camera_name="Border Gate",
            event_type="INTRUSION", severity="HIGH", timestamp=12.5,
            description="Person entered restricted zone.", source_type="video",
            track_id=3, identity="UNKNOWN", zone_id="z1", zone_name="Checkpoint",
            metadata={"object_type": "person"},
        )
        self.assertEqual(saved.id, "evt_1")

        fetched = self.events.get("evt_1")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.camera_id, "CAM-01")
        self.assertEqual(fetched.event_type, "INTRUSION")
        self.assertEqual(fetched.track_id, 3)
        self.assertEqual(fetched.identity, "UNKNOWN")
        self.assertEqual(fetched.metadata, {"object_type": "person"})

    def test_get_missing_event_returns_none(self):
        self.assertIsNone(self.events.get("does-not-exist"))

    def test_query_filters_by_camera(self):
        self.events.save(event_id="e1", camera_id="CAM-01", camera_name="A", event_type="INTRUSION", severity="HIGH", timestamp=1, description="d")
        self.events.save(event_id="e2", camera_id="CAM-02", camera_name="B", event_type="INTRUSION", severity="HIGH", timestamp=1, description="d")
        results = self.events.query(camera_id="CAM-01")
        self.assertEqual({r.id for r in results}, {"e1"})

    def test_query_filters_by_event_type_and_severity(self):
        self.events.save(event_id="e1", camera_id="CAM-01", camera_name="A", event_type="LOITERING", severity="MEDIUM", timestamp=1, description="d")
        self.events.save(event_id="e2", camera_id="CAM-01", camera_name="A", event_type="INTRUSION", severity="HIGH", timestamp=1, description="d")
        self.assertEqual({r.id for r in self.events.query(event_type="LOITERING")}, {"e1"})
        self.assertEqual({r.id for r in self.events.query(severity="HIGH")}, {"e2"})

    def test_query_filters_by_identity(self):
        self.events.save(event_id="e1", camera_id="CAM-01", camera_name="A", event_type="INTRUSION", severity="HIGH", timestamp=1, description="d", identity="Shreyas_Chavan")
        self.events.save(event_id="e2", camera_id="CAM-01", camera_name="A", event_type="INTRUSION", severity="HIGH", timestamp=1, description="d", identity="UNKNOWN")
        self.assertEqual({r.id for r in self.events.query(identity="Shreyas_Chavan")}, {"e1"})

    def test_query_time_range(self):
        self.events.save(event_id="e1", camera_id="CAM-01", camera_name="A", event_type="INTRUSION", severity="HIGH", timestamp=1, description="d")
        all_events = self.events.query()
        created = all_events[0].created_at
        self.assertEqual(len(self.events.query(start_time=created)), 1)
        self.assertEqual(len(self.events.query(start_time="9999-01-01T00:00:00")), 0)

    def test_query_pagination_never_exceeds_hard_cap(self):
        for i in range(5):
            self.events.save(event_id=f"e{i}", camera_id="CAM-01", camera_name="A", event_type="INTRUSION", severity="HIGH", timestamp=i, description="d")
        self.assertEqual(len(self.events.query(limit=2)), 2)
        self.assertEqual(len(self.events.query(limit=100000)), 5)  # clamped, but only 5 rows exist
        limited = self.events.query(limit=3, offset=0)
        next_page = self.events.query(limit=3, offset=3)
        self.assertEqual(len(limited), 3)
        self.assertEqual(len(next_page), 2)
        self.assertEqual({e.id for e in limited} & {e.id for e in next_page}, set())

    def test_query_orders_newest_first(self):
        self.events.save(event_id="e1", camera_id="CAM-01", camera_name="A", event_type="INTRUSION", severity="HIGH", timestamp=1, description="d")
        self.events.save(event_id="e2", camera_id="CAM-01", camera_name="A", event_type="INTRUSION", severity="HIGH", timestamp=2, description="d")
        results = self.events.query()
        self.assertEqual(results[0].id, "e2")  # most recently inserted first

    def test_related_to_same_camera_and_track(self):
        self.events.save(event_id="e1", camera_id="CAM-03", camera_name="A", event_type="INTRUSION", severity="HIGH", timestamp=1, description="d", track_id=17)
        self.events.save(event_id="e2", camera_id="CAM-03", camera_name="A", event_type="LOITERING", severity="MEDIUM", timestamp=2, description="d", track_id=17)
        self.events.save(event_id="e3", camera_id="CAM-03", camera_name="A", event_type="INTRUSION", severity="HIGH", timestamp=3, description="d", track_id=99)  # different track
        related = self.events.related_to("CAM-03", 17, exclude_event_id="e1")
        self.assertEqual({r.id for r in related}, {"e2"})

    def test_for_identity_aggregates_across_cameras(self):
        self.events.save(event_id="e1", camera_id="CAM-01", camera_name="A", event_type="INTRUSION", severity="HIGH", timestamp=1, description="d", identity="Shreyas_Chavan")
        self.events.save(event_id="e2", camera_id="CAM-02", camera_name="B", event_type="INTRUSION", severity="HIGH", timestamp=1, description="d", identity="Shreyas_Chavan")
        self.assertEqual({r.id for r in self.events.for_identity("Shreyas_Chavan")}, {"e1", "e2"})

    def test_query_by_alert_status_joins_correctly(self):
        self.events.save(event_id="e1", camera_id="CAM-01", camera_name="A", event_type="INTRUSION", severity="HIGH", timestamp=1, description="d")
        self.alerts.save(alert_id="a1", event_id="e1", camera_id="CAM-01", camera_name="A", severity="HIGH", title="t", message="m")
        self.assertEqual(len(self.events.query(status="NEW")), 1)
        self.assertEqual(len(self.events.query(status="RESOLVED")), 0)


class TestAlertRepository(StorageTestCase):
    def _make_event_and_alert(self, event_id="e1", alert_id="a1"):
        self.events.save(event_id=event_id, camera_id="CAM-01", camera_name="Border Gate", event_type="INTRUSION", severity="HIGH", timestamp=1, description="d")
        return self.alerts.save(alert_id=alert_id, event_id=event_id, camera_id="CAM-01", camera_name="Border Gate", severity="HIGH", title="INTRUSION", message="m")

    def test_new_alert_status_is_new(self):
        alert = self._make_event_and_alert()
        self.assertEqual(alert.status, "NEW")
        self.assertIsNone(alert.acknowledged_at)
        self.assertIsNone(alert.resolved_at)

    def test_new_to_acknowledged(self):
        self._make_event_and_alert()
        updated = self.alerts.transition("a1", "ACKNOWLEDGED")
        self.assertEqual(updated.status, "ACKNOWLEDGED")
        self.assertIsNotNone(updated.acknowledged_at)
        self.assertIsNone(updated.resolved_at)

    def test_new_to_resolved_directly(self):
        self._make_event_and_alert()
        updated = self.alerts.transition("a1", "RESOLVED")
        self.assertEqual(updated.status, "RESOLVED")
        self.assertIsNotNone(updated.resolved_at)

    def test_acknowledged_to_resolved(self):
        self._make_event_and_alert()
        self.alerts.transition("a1", "ACKNOWLEDGED")
        updated = self.alerts.transition("a1", "RESOLVED")
        self.assertEqual(updated.status, "RESOLVED")

    def test_resolved_to_anything_rejected(self):
        self._make_event_and_alert()
        self.alerts.transition("a1", "RESOLVED")
        with self.assertRaises(InvalidAlertTransition):
            self.alerts.transition("a1", "ACKNOWLEDGED")
        with self.assertRaises(InvalidAlertTransition):
            self.alerts.transition("a1", "NEW")

    def test_acknowledged_to_new_rejected(self):
        self._make_event_and_alert()
        self.alerts.transition("a1", "ACKNOWLEDGED")
        with self.assertRaises(InvalidAlertTransition):
            self.alerts.transition("a1", "NEW")

    def test_new_to_new_rejected(self):
        self._make_event_and_alert()
        with self.assertRaises(InvalidAlertTransition):
            self.alerts.transition("a1", "NEW")

    def test_transition_unknown_alert_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.alerts.transition("does-not-exist", "ACKNOWLEDGED")

    def test_query_filters_by_status(self):
        self._make_event_and_alert("e1", "a1")
        self._make_event_and_alert("e2", "a2")
        self.alerts.transition("a1", "ACKNOWLEDGED")
        self.assertEqual({a.id for a in self.alerts.query(status="ACKNOWLEDGED")}, {"a1"})
        self.assertEqual({a.id for a in self.alerts.query(status="NEW")}, {"a2"})

    def test_query_filters_by_event_type_via_join(self):
        self.events.save(event_id="e1", camera_id="CAM-01", camera_name="A", event_type="LOITERING", severity="MEDIUM", timestamp=1, description="d")
        self.alerts.save(alert_id="a1", event_id="e1", camera_id="CAM-01", camera_name="A", severity="MEDIUM", title="t", message="m")
        self.assertEqual(len(self.alerts.query(event_type="LOITERING")), 1)
        self.assertEqual(len(self.alerts.query(event_type="INTRUSION")), 0)


class TestZoneRepository(StorageTestCase):
    def test_create_and_get_zone(self):
        z = self.zones.create("z1", "CAM-01", "Checkpoint", "restricted", [[0, 0], [10, 0], [10, 10], [0, 10]])
        self.assertEqual(z.id, "z1")
        fetched = self.zones.get("z1")
        self.assertEqual(fetched.name, "Checkpoint")
        self.assertEqual(fetched.polygon, [(0, 0), (10, 0), (10, 10), (0, 10)])
        self.assertTrue(fetched.enabled)

    def test_create_rejects_invalid_type(self):
        with self.assertRaises(ValueError):
            self.zones.create("z1", "CAM-01", "Bad", "dangerous", [[0, 0], [1, 0], [1, 1]])

    def test_create_rejects_degenerate_polygon(self):
        with self.assertRaises(ValueError):
            self.zones.create("z1", "CAM-01", "TooFew", "restricted", [[0, 0], [1, 1]])

    def test_create_rejects_non_numeric_point(self):
        with self.assertRaises(ValueError):
            self.zones.create("z1", "CAM-01", "Bad", "restricted", [[0, 0], ["x", 1], [1, 1]])

    def test_create_rejects_negative_coordinates(self):
        with self.assertRaises(ValueError):
            self.zones.create("z1", "CAM-01", "Bad", "restricted", [[-1, 0], [1, 0], [1, 1]])

    def test_create_rejects_zero_area_polygon_all_points_identical(self):
        # A degenerate draw (e.g. a click with no drag) must never persist a
        # zone with no real geometry.
        with self.assertRaises(ValueError):
            self.zones.create("z1", "CAM-01", "Bad", "restricted", [[25, 98], [25, 98], [25, 98], [25, 98]])

    def test_create_rejects_zero_area_polygon_collinear_points(self):
        with self.assertRaises(ValueError):
            self.zones.create("z1", "CAM-01", "Bad", "restricted", [[0, 5], [10, 5], [20, 5]])

    def test_update_name_and_enabled(self):
        self.zones.create("z1", "CAM-01", "Original", "restricted", [[0, 0], [1, 0], [1, 1]])
        updated = self.zones.update("z1", name="Renamed", enabled=False)
        self.assertEqual(updated.name, "Renamed")
        self.assertFalse(updated.enabled)
        self.assertEqual(updated.type, "restricted")  # unchanged

    def test_update_polygon(self):
        self.zones.create("z1", "CAM-01", "Z", "restricted", [[0, 0], [1, 0], [1, 1]])
        updated = self.zones.update("z1", polygon=[[0, 0], [5, 0], [5, 5], [0, 5]])
        self.assertEqual(updated.polygon, [(0, 0), (5, 0), (5, 5), (0, 5)])

    def test_update_unknown_zone_raises(self):
        with self.assertRaises(ValueError):
            self.zones.update("does-not-exist", name="x")

    def test_delete_zone(self):
        self.zones.create("z1", "CAM-01", "Z", "restricted", [[0, 0], [1, 0], [1, 1]])
        self.assertTrue(self.zones.delete("z1"))
        self.assertIsNone(self.zones.get("z1"))

    def test_delete_missing_zone_returns_false(self):
        self.assertFalse(self.zones.delete("does-not-exist"))

    def test_list_for_camera_only_returns_that_cameras_zones(self):
        self.zones.create("z1", "CAM-01", "A", "restricted", [[0, 0], [1, 0], [1, 1]])
        self.zones.create("z2", "CAM-02", "B", "restricted", [[0, 0], [1, 0], [1, 1]])
        self.assertEqual({z.id for z in self.zones.list_for_camera("CAM-01")}, {"z1"})

    def test_list_for_camera_with_no_zones_is_empty(self):
        # The "NO ZONE CONFIGURED" case — never a fake zone.
        self.assertEqual(self.zones.list_for_camera("CAM-99"), [])

    def test_seed_if_empty_only_seeds_once(self):
        class FakeZone:
            def __init__(self, id, name, zone_type, polygon):
                self.id, self.name, self.zone_type, self.polygon = id, name, zone_type, polygon

        seed = [FakeZone("z1", "Checkpoint", "restricted", [(0, 0), (1, 0), (1, 1)])]
        self.zones.seed_if_empty("CAM-01", seed)
        self.assertEqual(len(self.zones.list_for_camera("CAM-01")), 1)

        # Operator renames the seeded zone...
        self.zones.update("z1", name="Renamed By Operator")
        # ...a second seed attempt (e.g. camera restart) must not clobber it.
        self.zones.seed_if_empty("CAM-01", seed)
        zones = self.zones.list_for_camera("CAM-01")
        self.assertEqual(len(zones), 1)
        self.assertEqual(zones[0].name, "Renamed By Operator")


class TestEventPersistenceService(StorageTestCase):
    def setUp(self):
        super().setUp()
        self.snapshot_dir = os.path.join(self._tmp_dir, "snapshots")
        self.service = EventPersistenceService(self.events, self.alerts, snapshot_dir=self.snapshot_dir)

    def _fake_event_and_alert(self, event_id="evt_1", alert_id="alr_1"):
        evt = SecurityEvent(
            event_id=event_id, event_type=EventType.INTRUSION, severity=Severity.HIGH,
            camera_id="CAM-01", track_id=7, timestamp=1.0, zone_id="z1",
            message="Person entered restricted zone.",
            metadata={"identity": "UNKNOWN", "zone_name": "Checkpoint"},
        )
        alr = Alert(
            alert_id=alert_id, event_id=event_id, severity=Severity.HIGH,
            title="INTRUSION DETECTED", message="Person entered restricted zone.",
            timestamp=1.0, status=AlertStatus.NEW,
        )
        return evt, alr

    def test_record_persists_event_and_alert_with_snapshot(self):
        evt, alr = self._fake_event_and_alert()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        self.service.record("CAM-01", "Border Gate", "video", [evt], [alr], frame)

        stored_event = self.events.get("evt_1")
        self.assertIsNotNone(stored_event)
        self.assertEqual(stored_event.identity, "UNKNOWN")
        self.assertEqual(stored_event.zone_name, "Checkpoint")
        self.assertIsNotNone(stored_event.snapshot_path)
        self.assertTrue(os.path.exists(stored_event.snapshot_path))

        stored_alert = self.alerts.get("alr_1")
        self.assertIsNotNone(stored_alert)
        self.assertEqual(stored_alert.status, "NEW")

    def test_record_without_frame_saves_event_with_no_snapshot(self):
        # A missing/unbuildable annotated frame must not block persistence
        # of the event itself — an honest "no snapshot" beats losing the event.
        evt, alr = self._fake_event_and_alert()
        self.service.record("CAM-01", "Border Gate", "video", [evt], [alr], None)
        stored_event = self.events.get("evt_1")
        self.assertIsNotNone(stored_event)
        self.assertIsNone(stored_event.snapshot_path)

    def test_record_survives_repository_failure_without_raising(self):
        evt, alr = self._fake_event_and_alert()

        class BrokenEventRepo:
            def save(self, *a, **kw):
                raise RuntimeError("disk full")

        broken_service = EventPersistenceService(BrokenEventRepo(), self.alerts, snapshot_dir=self.snapshot_dir)
        try:
            broken_service.record("CAM-01", "Border Gate", "video", [evt], [alr], None)
        except Exception as e:
            self.fail(f"record() must never raise, even on repository failure: {e}")


if __name__ == "__main__":
    unittest.main()
