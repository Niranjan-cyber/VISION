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
    serialize_event,
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

    def test_serialize_alert_includes_track_id_when_provided(self):
        alr = Alert(
            alert_id="alr_1", event_id="evt_1", severity=Severity.HIGH,
            title="INTRUSION", message="msg", timestamp=1.5,
            status=AlertStatus.NEW, metadata={"object_type": "person"},
        )
        result = serialize_alert(alr, track_id=7)
        self.assertEqual(result["track_id"], 7)
        self.assertEqual(result["object_type"], "person")


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


if __name__ == "__main__":
    unittest.main()
