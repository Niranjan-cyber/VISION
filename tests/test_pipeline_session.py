import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.session import PipelineSession


class TestPipelineSessionGoldenDemo(unittest.TestCase):
    """
    End-to-end smoke test for the extracted pipeline session against the
    actual golden demo video + zone config used for the SIH dashboard. This
    is the one test that would have caught the pre-fix bugs found during the
    MVP audit (empty default video, zone/video mismatch producing zero
    events, the face_crop NameError) — none of the other 129 tests exercise
    the assembled pipeline end-to-end.
    """

    @classmethod
    def setUpClass(cls):
        cls.session = PipelineSession(
            video_path="data/videos/shreyas1.mp4",
            zones_path="configs/zones_demo.yaml",
            loitering_duration=3.0,
            enable_anpr=False,
            verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        cls.session.release()

    def test_status_reflects_real_subsystem_state(self):
        status = self.session.status
        self.assertTrue(status["video"])
        self.assertTrue(status["detection"])
        self.assertTrue(status["tracking"])
        self.assertTrue(status["face_id"])
        self.assertFalse(status["anpr"])  # explicitly disabled for this session
        self.assertTrue(status["events"])  # zones_path was supplied

    def test_golden_video_produces_nonempty_output_and_fires_events(self):
        session = self.session
        frame_count = 0
        while True:
            frame = session.source.read_frame()
            if frame is None:
                break
            frame_count += 1
            session.process_frame(frame, session.source.current_frame)

        self.assertGreater(frame_count, 0, "golden demo video must contain frames")
        self.assertGreater(len(session.observed_unique_track_ids), 0, "expected at least one tracked person")
        self.assertGreater(session.total_faces_detected, 0, "expected faces to be detected in the golden clip")
        self.assertGreater(session.total_recognized_faces, 0, "expected the enrolled identity to be recognized")

        # The calibrated golden zone must actually fire events on this video —
        # this is the exact class of failure the pre-fix configs/zones.yaml
        # silently had (zero events on every shipped demo video).
        self.assertIsNotNone(session.event_engine)
        event_types = {ev.event_type.value for ev in session.event_engine.event_history}
        self.assertIn("INTRUSION", event_types)
        self.assertIn("LOITERING", event_types)


if __name__ == "__main__":
    unittest.main()
