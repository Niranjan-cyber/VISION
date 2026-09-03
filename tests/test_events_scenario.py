import unittest
from src.core.types import BoundingBox
from src.events.engine import EventEngine
from src.events.state import ObjectState
from src.events.types import EventType, Severity
from src.events.zone import Zone


class TestEventsScenario(unittest.TestCase):
    """
    End-to-End Scenario Test for Vertical Slice 7 (Specification Section 28):
    Person #17 approaches restricted polygon, crosses boundary, triggers INTRUSION + UNKNOWN_PERSON_INTRUSION,
    dwells inside for >= 30 seconds, and triggers LOITERING.
    """

    def test_section_28_verification_scenario(self):
        # 1. Setup border restricted polygon: [100, 100] to [500, 450]
        zone = Zone(
            id="restricted_zone_1",
            name="Border Restricted Area",
            zone_type="restricted",
            polygon=[(100, 100), (500, 100), (500, 450), (100, 450)],
        )
        engine = EventEngine(
            zones=[zone],
            loitering_duration=30.0,
        )

        camera = "BOP-03"

        # Step 1: Person #17 approaches restricted polygon from outside
        # Position: bottom-center at (50, 50) -> OUTSIDE
        state_f1 = ObjectState(
            track_id=17,
            object_type="person",
            bbox=BoundingBox(x1=30, y1=10, x2=70, y2=50),
            confidence=0.92,
            camera_id=camera,
            has_face_detected=True,
            identity="UNKNOWN",
            face_similarity=0.48,
        )
        events_f1, alerts_f1 = engine.update([state_f1], timestamp=0.0)
        self.assertEqual(len(events_f1), 0, "No event should trigger while outside")
        self.assertEqual(len(alerts_f1), 0)

        # Step 2: Person #17 crosses the boundary into the restricted zone
        # Position: bottom-center at (200, 200) -> INSIDE
        state_f2 = ObjectState(
            track_id=17,
            object_type="person",
            bbox=BoundingBox(x1=180, y1=140, x2=220, y2=200),
            confidence=0.94,
            camera_id=camera,
            has_face_detected=True,
            identity="UNKNOWN",
            face_similarity=0.48,
        )
        events_f2, alerts_f2 = engine.update([state_f2], timestamp=1.0)

        # Must generate INTRUSION and UNKNOWN_PERSON_INTRUSION
        self.assertEqual(len(events_f2), 2, "Expected INTRUSION and UNKNOWN_PERSON_INTRUSION")
        event_types = {e.event_type for e in events_f2}
        self.assertIn(EventType.INTRUSION, event_types)
        self.assertIn(EventType.UNKNOWN_PERSON_INTRUSION, event_types)

        for evt in events_f2:
            self.assertEqual(evt.track_id, 17)
            self.assertEqual(evt.severity, Severity.HIGH)
            self.assertEqual(evt.camera_id, camera)
            self.assertEqual(evt.zone_id, "restricted_zone_1")
            self.assertEqual(evt.metadata["identity"], "UNKNOWN")

        # Step 3: Person #17 stays inside at t = 10s and t = 29s (no duplicates)
        state_f3 = ObjectState(
            track_id=17,
            object_type="person",
            bbox=BoundingBox(x1=182, y1=142, x2=222, y2=202),
            confidence=0.94,
            camera_id=camera,
            has_face_detected=True,
            identity="UNKNOWN",
        )
        events_f3, _ = engine.update([state_f3], timestamp=10.0)
        self.assertEqual(len(events_f3), 0, "No duplicate intrusion event on continuous dwell")

        events_f4, _ = engine.update([state_f3], timestamp=30.0)  # dwell = 30 - 1 = 29s
        self.assertEqual(len(events_f4), 0, "Loitering should not trigger before 30 seconds")

        # Step 4: Person #17 reaches dwell >= 30 seconds (t = 31.0s, dwell = 30.0s)
        events_f5, alerts_f5 = engine.update([state_f3], timestamp=31.0)
        self.assertEqual(len(events_f5), 1, "LOITERING event must be generated")
        loitering_evt = events_f5[0]
        self.assertEqual(loitering_evt.event_type, EventType.LOITERING)
        self.assertEqual(loitering_evt.severity, Severity.MEDIUM)
        self.assertEqual(loitering_evt.track_id, 17)
        self.assertEqual(loitering_evt.zone_id, "restricted_zone_1")
        self.assertGreaterEqual(loitering_evt.metadata["duration"], 30.0)

        self.assertEqual(len(alerts_f5), 1)
        self.assertEqual(alerts_f5[0].title, "⚠️ LOITERING DETECTED")

        # Step 5: Continued dwell at t = 35.0s does not generate duplicate LOITERING
        events_f6, _ = engine.update([state_f3], timestamp=35.0)
        self.assertEqual(len(events_f6), 0, "No duplicate loitering events while inside")


if __name__ == "__main__":
    unittest.main()
