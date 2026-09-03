import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.types import BoundingBox
from src.events.state import ObjectState
from src.events.types import Alert, AlertStatus, EventType, SecurityEvent, Severity
from src.pipeline.serialize import (
    serialize_alert,
    serialize_camera_summary,
    serialize_event,
    serialize_global_detections,
    serialize_global_events,
    serialize_global_status,
    serialize_person,
    serialize_state,
    serialize_vehicle,
)


class TestSerializePerson(unittest.TestCase):
    def test_recognized_identity(self):
        obj = ObjectState(
            track_id=1,
            object_type="person",
            bbox=BoundingBox(x1=10, y1=20, x2=110, y2=220),
            confidence=0.91,
            identity="PERSON_A",
            face_similarity=0.7123,
            has_face_detected=True,
            current_zone="restricted_area",
        )
        result = serialize_person(obj)
        self.assertEqual(result["track_id"], 1)
        self.assertEqual(result["identity"], "PERSON_A")
        self.assertEqual(result["face_similarity"], 0.712)  # rounded
        self.assertEqual(result["bbox"], [10, 20, 110, 220])
        self.assertEqual(result["zone"], "restricted_area")

    def test_unknown_face_vs_no_face_detected_are_distinct(self):
        no_face = ObjectState(
            track_id=2, object_type="person", bbox=BoundingBox(0, 0, 10, 10),
            confidence=0.9, identity=None, has_face_detected=False,
        )
        unmatched_face = ObjectState(
            track_id=3, object_type="person", bbox=BoundingBox(0, 0, 10, 10),
            confidence=0.9, identity="UNKNOWN", face_similarity=0.3, has_face_detected=True,
        )
        self.assertIsNone(serialize_person(no_face)["identity"])
        self.assertEqual(serialize_person(unmatched_face)["identity"], "UNKNOWN")


class TestSerializeVehicle(unittest.TestCase):
    def setUp(self):
        self.obj = ObjectState(
            track_id=4,
            object_type="car",
            bbox=BoundingBox(0, 0, 50, 30),
            confidence=0.85,
            plate="MH12AB1234",
            plate_confidence=0.92,
        )

    def test_plate_included_when_anpr_enabled(self):
        result = serialize_vehicle(self.obj, anpr_enabled=True)
        self.assertEqual(result["plate"], "MH12AB1234")
        self.assertEqual(result["plate_confidence"], 0.92)

    def test_plate_absent_never_fake_when_anpr_disabled(self):
        result = serialize_vehicle(self.obj, anpr_enabled=False)
        self.assertNotIn("plate", result)
        self.assertNotIn("plate_confidence", result)


class TestSerializeEventsAndAlerts(unittest.TestCase):
    def test_serialize_event(self):
        evt = SecurityEvent(
            event_id="evt_1", event_type=EventType.INTRUSION, severity=Severity.HIGH,
            camera_id="BOP-01", track_id=1, timestamp=1.5, zone_id="z1",
            message="msg", metadata={"zone_name": "Checkpoint"},
        )
        result = serialize_event(evt)
        self.assertEqual(result["event_type"], "INTRUSION")
        self.assertEqual(result["severity"], "HIGH")
        self.assertEqual(result["zone_name"], "Checkpoint")

    def test_serialize_alert_includes_camera_and_track_id(self):
        alr = Alert(
            alert_id="alr_1", event_id="evt_1", severity=Severity.HIGH,
            title="INTRUSION", message="msg", timestamp=1.5,
            status=AlertStatus.NEW, metadata={"object_type": "person"},
        )
        result = serialize_alert(alr, "CAM-02", "BOP East", track_id=7)
        self.assertEqual(result["track_id"], 7)
        self.assertEqual(result["object_type"], "person")
        self.assertEqual(result["camera_id"], "CAM-02")
        self.assertEqual(result["camera_name"], "BOP East")


class TestSerializeState(unittest.TestCase):
    def test_full_state_contract_shape(self):
        session = MagicMock()
        session.latest_object_states = [
            ObjectState(track_id=1, object_type="person", bbox=BoundingBox(0, 0, 10, 10), confidence=0.9),
        ]
        session.enable_anpr = False
        session.latest_faces = []
        session.track_identity_cache = {}
        session.event_engine = None
        session.frame_index = 5
        session.status = {"video": True, "detection": True, "tracking": True, "face_id": True, "anpr": False, "events": False}

        state = serialize_state(session, result=None)
        for key in ("timestamp", "frame_id", "persons", "vehicles", "statistics", "anpr_enabled", "status"):
            self.assertIn(key, state)
        for key in ("persons", "vehicles", "faces_detected", "recognized_faces", "active_events"):
            self.assertIn(key, state["statistics"])
        self.assertEqual(state["anpr_enabled"], False)


def _make_mock_camera(camera_id, camera_name, status="online", error=None,
                       video_path="data/videos/x.mp4", zones_path=None,
                       persons=None, enable_anpr=False, alerts=None):
    """Builds a MagicMock CameraSession good enough to drive the
    multi-camera serializers without spinning up a real PipelineSession."""
    cam = MagicMock()
    cam.camera_id = camera_id
    cam.camera_name = camera_name
    cam.status = status
    cam.error = error
    cam.config.video_path = video_path
    cam.config.zones_path = zones_path
    cam.config.enable_anpr = enable_anpr

    if status in ("online",):
        session = MagicMock()
        session.enable_anpr = enable_anpr
        session.latest_object_states = persons or []
        session.latest_faces = [o for o in (persons or []) if o.has_face_detected]
        session.track_identity_cache = {}
        session.frame_index = 1
        session.status = {"video": True, "detection": True, "tracking": True, "face_id": True, "anpr": enable_anpr, "events": zones_path is not None}
        if alerts:
            engine = MagicMock()
            engine.event_history = alerts
            engine.active_alerts = alerts
            session.event_engine = engine
        else:
            session.event_engine = None
        cam.snapshot.return_value = (session, None)
    else:
        cam.snapshot.return_value = (None, None)
    return cam


class TestCameraSerializers(unittest.TestCase):
    def test_serialize_camera_summary_online(self):
        cam = _make_mock_camera("CAM-01", "Border Gate", status="online", zones_path="configs/zones_demo.yaml")
        result = serialize_camera_summary(cam)
        self.assertEqual(result["camera_id"], "CAM-01")
        self.assertEqual(result["camera_name"], "Border Gate")
        self.assertEqual(result["status"], "online")
        self.assertIsNone(result["error"])
        self.assertIn("statistics", result)

    def test_serialize_camera_summary_error_state_has_no_fake_stats(self):
        cam = _make_mock_camera("CAM-05", "Broken Cam", status="error", error="failed to start: bad path")
        result = serialize_camera_summary(cam)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "failed to start: bad path")
        self.assertEqual(result["statistics"]["persons"], 0)

    def test_serialize_global_status_aggregates_all_cameras(self):
        manager = MagicMock()
        manager.list_cameras.return_value = [
            _make_mock_camera("CAM-01", "Border Gate", status="online"),
            _make_mock_camera("CAM-02", "BOP East", status="error", error="boom"),
        ]
        result = serialize_global_status(manager)
        self.assertEqual(result["cameras_active"], 2)
        self.assertEqual(result["cameras_max"], 4)
        self.assertEqual(result["cameras"]["CAM-01"]["status"], "online")
        self.assertEqual(result["cameras"]["CAM-02"]["error"], "boom")

    def test_serialize_global_detections_sums_real_per_camera_stats(self):
        p1 = ObjectState(track_id=1, object_type="person", bbox=BoundingBox(0, 0, 10, 10), confidence=0.9, has_face_detected=True)
        p2 = ObjectState(track_id=2, object_type="person", bbox=BoundingBox(0, 0, 10, 10), confidence=0.9, has_face_detected=False)
        manager = MagicMock()
        manager.list_cameras.return_value = [
            _make_mock_camera("CAM-01", "Border Gate", persons=[p1]),
            _make_mock_camera("CAM-02", "BOP East", persons=[p1, p2]),
        ]
        result = serialize_global_detections(manager)
        self.assertEqual(len(result["cameras"]), 2)
        self.assertEqual(result["statistics"]["persons"], 3)  # 1 + 2, not invented
        self.assertEqual(result["statistics"]["cameras_active"], 2)
        self.assertEqual(result["cameras"][0]["camera_id"], "CAM-01")

    def test_serialize_global_events_merges_and_sorts_by_time(self):
        evt_early = SecurityEvent(event_id="e1", event_type=EventType.INTRUSION, severity=Severity.HIGH,
                                   camera_id="CAM-01", track_id=1, timestamp=1.0, zone_id="z1", message="m", metadata={})
        evt_late = SecurityEvent(event_id="e2", event_type=EventType.LOITERING, severity=Severity.MEDIUM,
                                  camera_id="CAM-02", track_id=1, timestamp=5.0, zone_id="z1", message="m", metadata={})
        alr_early = Alert(alert_id="a1", event_id="e1", severity=Severity.HIGH, title="INTRUSION", message="m",
                           timestamp=1.0, status=AlertStatus.NEW, metadata={})
        alr_late = Alert(alert_id="a2", event_id="e2", severity=Severity.MEDIUM, title="LOITERING", message="m",
                          timestamp=5.0, status=AlertStatus.NEW, metadata={})

        manager = MagicMock()
        manager.list_cameras.return_value = [
            _make_mock_camera("CAM-01", "Border Gate", alerts=[evt_early] if False else [alr_early]),
            _make_mock_camera("CAM-02", "BOP East", alerts=[alr_late]),
        ]
        # Wire event_history separately so serialize_events' track_id lookup works.
        for cam, evt in zip(manager.list_cameras.return_value, [evt_early, evt_late]):
            session, _ = cam.snapshot()
            session.event_engine.event_history = [evt]

        result = serialize_global_events(manager)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["camera_id"], "CAM-02")  # newest first
        self.assertEqual(result[1]["camera_id"], "CAM-01")


if __name__ == "__main__":
    unittest.main()
