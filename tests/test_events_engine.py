import unittest
from src.core.types import BoundingBox
from src.events.engine import EventEngine
from src.events.state import ObjectState
from src.events.types import AlertStatus, EventType, Severity
from src.events.zone import Zone


class TestEventsEngine(unittest.TestCase):
    """Unit test suite for EventEngine rule evaluation, state tracking, and deduplication."""

    def setUp(self):
        # Restricted polygon: (100, 100) to (300, 300)
        self.restricted_zone = Zone(
            id="zone_restricted",
            name="Restricted Area",
            zone_type="restricted",
            polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        )

        self.engine = EventEngine(
            zones=[self.restricted_zone],
            loitering_duration=30.0,
            stationary_duration=60.0,
            movement_threshold=15.0,
        )

    def test_1_zone_transitions_and_deduplication(self):
        """Transition outside -> inside produces INTRUSION; inside -> inside does not duplicate."""
        # Frame 1: Person outside at (50, 50)
        p1 = ObjectState(
            track_id=1,
            object_type="person",
            bbox=BoundingBox(x1=40, y1=20, x2=60, y2=50),
            confidence=0.9,
        )
        events, alerts = self.engine.update([p1], timestamp=10.0)
        self.assertEqual(len(events), 0)
        self.assertEqual(len(alerts), 0)

        # Frame 2: Person steps inside at (150, 150) -> INTRUSION
        p2 = ObjectState(
            track_id=1,
            object_type="person",
            bbox=BoundingBox(x1=140, y1=100, x2=160, y2=150),
            confidence=0.9,
        )
        events, alerts = self.engine.update([p2], timestamp=11.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(events[0].event_type, EventType.INTRUSION)
        self.assertEqual(events[0].severity, Severity.HIGH)
        self.assertEqual(events[0].track_id, 1)
        self.assertEqual(events[0].zone_id, "zone_restricted")

        # Frame 3: Person stays inside at (160, 160) -> NO duplicate event
        p3 = ObjectState(
            track_id=1,
            object_type="person",
            bbox=BoundingBox(x1=150, y1=110, x2=170, y2=160),
            confidence=0.9,
        )
        events, alerts = self.engine.update([p3], timestamp=12.0)
        self.assertEqual(len(events), 0)
        self.assertEqual(len(alerts), 0)

        # Frame 4: Person exits to (50, 50)
        p4 = ObjectState(
            track_id=1,
            object_type="person",
            bbox=BoundingBox(x1=40, y1=20, x2=60, y2=50),
            confidence=0.9,
        )
        events, alerts = self.engine.update([p4], timestamp=15.0)
        self.assertEqual(len(events), 0)

        # Frame 5: Person re-enters at (200, 200) -> NEW INTRUSION generated
        p5 = ObjectState(
            track_id=1,
            object_type="person",
            bbox=BoundingBox(x1=180, y1=150, x2=220, y2=200),
            confidence=0.9,
        )
        events, alerts = self.engine.update([p5], timestamp=20.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.INTRUSION)

    def test_2_identity_distinction_unknown_vs_no_face(self):
        """Crucial distinction: missing face detection != unknown identity."""
        # Case A: Person enters with NO face detection (has_face_detected=False)
        p_noface = ObjectState(
            track_id=10,
            object_type="person",
            bbox=BoundingBox(x1=140, y1=100, x2=160, y2=150),
            confidence=0.9,
            has_face_detected=False,
            identity=None,
        )
        events, _ = self.engine.update([p_noface], timestamp=1.0)
        # Should generate standard INTRUSION, but NOT UNKNOWN_PERSON_INTRUSION
        event_types = [e.event_type for e in events]
        self.assertIn(EventType.INTRUSION, event_types)
        self.assertNotIn(EventType.UNKNOWN_PERSON_INTRUSION, event_types)

        self.engine.reset()

        # Case B: Person enters with face detected but NOT matched (identity='UNKNOWN')
        p_unknown = ObjectState(
            track_id=11,
            object_type="person",
            bbox=BoundingBox(x1=140, y1=100, x2=160, y2=150),
            confidence=0.9,
            has_face_detected=True,
            identity="UNKNOWN",
            face_similarity=0.45,
        )
        events, _ = self.engine.update([p_unknown], timestamp=2.0)
        # Should generate BOTH INTRUSION and UNKNOWN_PERSON_INTRUSION
        event_types = [e.event_type for e in events]
        self.assertIn(EventType.INTRUSION, event_types)
        self.assertIn(EventType.UNKNOWN_PERSON_INTRUSION, event_types)

        self.engine.reset()

        # Case C: Person enters with face recognized as authorized personnel
        p_known = ObjectState(
            track_id=12,
            object_type="person",
            bbox=BoundingBox(x1=140, y1=100, x2=160, y2=150),
            confidence=0.9,
            has_face_detected=True,
            identity="Commander_Smith",
            face_similarity=0.78,
        )
        events, _ = self.engine.update([p_known], timestamp=3.0)
        # Should generate INTRUSION with identity metadata, but NOT UNKNOWN_PERSON_INTRUSION
        event_types = [e.event_type for e in events]
        self.assertIn(EventType.INTRUSION, event_types)
        self.assertNotIn(EventType.UNKNOWN_PERSON_INTRUSION, event_types)
        self.assertEqual(events[0].metadata["identity"], "Commander_Smith")

    def test_3_loitering_timing_and_reset(self):
        """Loitering rule triggers after continuous dwell time, suppresses duplicates, resets on exit."""
        # Entry at t=10.0s
        p = ObjectState(
            track_id=20,
            object_type="person",
            bbox=BoundingBox(x1=140, y1=100, x2=160, y2=150),
            confidence=0.9,
        )
        self.engine.update([p], timestamp=10.0)

        # t=20.0s (10s dwell) -> No loitering
        events, _ = self.engine.update([p], timestamp=20.0)
        self.assertEqual(len(events), 0)

        # t=39.0s (29s dwell) -> No loitering
        events, _ = self.engine.update([p], timestamp=39.0)
        self.assertEqual(len(events), 0)

        # t=40.0s (30s dwell >= loitering_duration) -> LOITERING event!
        events, alerts = self.engine.update([p], timestamp=40.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.LOITERING)
        self.assertEqual(events[0].severity, Severity.MEDIUM)
        self.assertEqual(alerts[0].title, "⚠️ LOITERING DETECTED")

        # t=45.0s (35s dwell) -> No duplicate loitering event
        events, _ = self.engine.update([p], timestamp=45.0)
        self.assertEqual(len(events), 0)

        # t=50.0s -> Exits zone
        p_exit = ObjectState(
            track_id=20,
            object_type="person",
            bbox=BoundingBox(x1=10, y1=10, x2=30, y2=50),
            confidence=0.9,
        )
        self.engine.update([p_exit], timestamp=50.0)

        # t=60.0s -> Re-enters zone
        self.engine.update([p], timestamp=60.0)

        # t=80.0s (20s into new entry) -> No loitering yet
        events, _ = self.engine.update([p], timestamp=80.0)
        self.assertEqual(len(events), 0)

        # t=90.0s (30s into new entry) -> NEW LOITERING event!
        events, _ = self.engine.update([p], timestamp=90.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.LOITERING)

    def test_4_suspicious_vehicle_rule(self):
        """Stationary vehicle triggers SUSPICIOUS_VEHICLE after threshold; moving vehicle does not."""
        # Vehicle enters at t=0.0s
        veh = ObjectState(
            track_id=30,
            object_type="car",
            bbox=BoundingBox(x1=150, y1=120, x2=250, y2=200),
            confidence=0.88,
            plate="DL01AB1234",
            plate_confidence=0.92,
        )
        # Entry produces vehicle INTRUSION
        events, _ = self.engine.update([veh], timestamp=0.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.INTRUSION)

        # Vehicle remains stationary (same bbox) at t=30.0s (< 60s)
        events, _ = self.engine.update([veh], timestamp=30.0)
        self.assertEqual(len(events), 0)

        # Vehicle remains stationary at t=60.0s (>= 60s) -> SUSPICIOUS_VEHICLE!
        events, alerts = self.engine.update([veh], timestamp=60.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.SUSPICIOUS_VEHICLE)
        self.assertEqual(events[0].severity, Severity.MEDIUM)
        self.assertEqual(events[0].metadata["plate"], "DL01AB1234")
        self.assertIn("DL01AB1234", alerts[0].message)

        # Vehicle moving significantly resets stationary timer
        veh_moved = ObjectState(
            track_id=30,
            object_type="car",
            bbox=BoundingBox(x1=180, y1=150, x2=280, y2=250),  # shifted by > 30px
            confidence=0.88,
        )
        events, _ = self.engine.update([veh_moved], timestamp=70.0)
        self.assertEqual(len(events), 0)

    def test_5_alert_representation_and_lifecycle(self):
        """Operational alerts are created in memory with status NEW."""
        p = ObjectState(
            track_id=40,
            object_type="person",
            bbox=BoundingBox(x1=140, y1=100, x2=160, y2=150),
            confidence=0.9,
            has_face_detected=True,
            identity="UNKNOWN",
        )
        _, alerts = self.engine.update([p], timestamp=5.0)
        self.assertTrue(len(alerts) >= 1)
        alert = alerts[0]
        self.assertEqual(alert.status, AlertStatus.NEW)
        self.assertTrue(len(alert.alert_id) > 0)
        self.assertTrue(len(alert.title) > 0)
        self.assertTrue(len(alert.message) > 0)
        self.assertIn("Camera:", alert.message)
        self.assertIn("Severity:", alert.message)


if __name__ == "__main__":
    unittest.main()
