"""
Multi-camera orchestration tests. These exercise real PipelineSession
instances (real models, real video files) inside CameraManager, not mocks —
this is the layer the Phase 2 multi-camera work actually adds, so it is
tested against the real thing rather than a stand-in for it.

Grouped into small classes with shared setUpClass fixtures to keep total
model-loading cost down (each real camera takes several seconds to spin up).
"""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.camera_manager import CameraLimitReached, CameraManager, MAX_ACTIVE_CAMERAS


class TestCameraLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manager = CameraManager()

    @classmethod
    def tearDownClass(cls):
        cls.manager.shutdown()

    def test_camera_created_with_generated_id_and_becomes_online(self):
        cam = self.manager.add_camera(
            camera_name="Border Gate",
            video_path="data/videos/shreyas1.mp4",
            zones_path="configs/zones_demo.yaml",
            loitering_duration=3.0,
        )
        self.assertEqual(cam.camera_id, "CAM-01")
        self.assertEqual(cam.camera_name, "Border Gate")

        deadline = time.time() + 30
        while time.time() < deadline and cam.status == "starting":
            time.sleep(0.5)
        self.assertEqual(cam.status, "online", f"error={cam.error}")

    def test_camera_removal_frees_its_slot(self):
        before = self.manager.active_count
        cam = self.manager.add_camera(camera_name="Temp", video_path="data/videos/test.mp4")
        self.assertEqual(self.manager.active_count, before + 1)

        removed = self.manager.remove_camera(cam.camera_id)
        self.assertTrue(removed)
        self.assertEqual(self.manager.active_count, before)
        self.assertIsNone(self.manager.get(cam.camera_id))

    def test_removing_unknown_camera_returns_false(self):
        self.assertFalse(self.manager.remove_camera("CAM-does-not-exist"))

    def test_restarting_unknown_camera_returns_false(self):
        self.assertFalse(self.manager.restart_camera("CAM-does-not-exist"))


class TestFourCameraLimit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manager = CameraManager()
        # Same small clip reused four times deliberately — this test is
        # about the count enforcement, not per-camera content.
        cls.cams = [
            cls.manager.add_camera(camera_name=f"Slot {i}", video_path="data/videos/test.mp4")
            for i in range(MAX_ACTIVE_CAMERAS)
        ]

    @classmethod
    def tearDownClass(cls):
        cls.manager.shutdown()

    def test_manager_holds_exactly_max_cameras(self):
        self.assertEqual(self.manager.active_count, MAX_ACTIVE_CAMERAS)

    def test_fifth_camera_rejected_with_exact_message(self):
        with self.assertRaises(CameraLimitReached) as ctx:
            self.manager.add_camera(camera_name="One Too Many", video_path="data/videos/test.mp4")
        self.assertEqual(str(ctx.exception), "Maximum 4 active camera streams reached.")
        self.assertEqual(self.manager.active_count, MAX_ACTIVE_CAMERAS)  # unchanged

    def test_removing_one_allows_a_new_camera(self):
        self.manager.remove_camera(self.cams[0].camera_id)
        self.assertEqual(self.manager.active_count, MAX_ACTIVE_CAMERAS - 1)

        new_cam = self.manager.add_camera(camera_name="Replacement", video_path="data/videos/test.mp4")
        self.assertEqual(self.manager.active_count, MAX_ACTIVE_CAMERAS)
        self.manager.remove_camera(new_cam.camera_id)  # keep count symmetric for other tests in this class


class TestCameraIsolation(unittest.TestCase):
    """Two cameras running two different videos/identities simultaneously —
    verifies track IDs, identity state, and event history never leak across
    camera boundaries (Phase 2's core correctness requirement)."""

    @classmethod
    def setUpClass(cls):
        cls.manager = CameraManager()
        cls.cam_a = cls.manager.add_camera(
            camera_name="Border Gate",
            video_path="data/videos/shreyas1.mp4",
            zones_path="configs/zones_demo.yaml",
            loitering_duration=3.0,
        )
        cls.cam_b = cls.manager.add_camera(
            camera_name="BOP East",
            video_path="data/videos/jaysingpure1.mp4",
            zones_path="configs/zones_cam02.yaml",
            loitering_duration=3.0,
        )
        cls._wait_online(cls.cam_a)
        cls._wait_online(cls.cam_b)
        cls._wait_for_identity(cls.cam_a, track_id=1)
        cls._wait_for_identity(cls.cam_b, track_id=1)
        time.sleep(1)  # small margin for the event engine to catch up on the latest frame

    @classmethod
    def _wait_online(cls, cam, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline and cam.status == "starting":
            time.sleep(0.5)

    @classmethod
    def _wait_for_identity(cls, cam, track_id, timeout=90):
        """Two cameras run concurrently in this test, competing for CPU —
        poll for the identity cache to actually populate rather than
        guessing a fixed sleep duration. Generous timeout because a full
        `unittest discover` run has much more accumulated system load by
        the time this class runs than an isolated run of just this file."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            session, _ = cam.snapshot()
            if session is not None and track_id in session.track_identity_cache:
                return
            time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.manager.shutdown()

    def test_both_cameras_online_independently(self):
        self.assertEqual(self.cam_a.status, "online", self.cam_a.error)
        self.assertEqual(self.cam_b.status, "online", self.cam_b.error)

    def test_track_id_1_refers_to_different_people_on_each_camera(self):
        session_a, _ = self.cam_a.snapshot()
        session_b, _ = self.cam_b.snapshot()
        match_a = session_a.track_identity_cache.get(1)
        match_b = session_b.track_identity_cache.get(1)
        self.assertIsNotNone(match_a, "CAM-A track #1 should have an identity match by now")
        self.assertIsNotNone(match_b, "CAM-B track #1 should have an identity match by now")
        self.assertEqual(match_a.identity, "Shreyas_Chavan")
        self.assertEqual(match_b.identity, "Atharva_Jaysingpure")
        self.assertNotEqual(match_a.identity, match_b.identity)

    def test_event_engines_are_separate_objects_with_separate_history(self):
        session_a, _ = self.cam_a.snapshot()
        session_b, _ = self.cam_b.snapshot()
        self.assertIsNot(session_a.event_engine, session_b.event_engine)
        self.assertIsNot(session_a.track_identity_cache, session_b.track_identity_cache)
        # Both should have independently fired their own INTRUSION by now.
        types_a = {e.event_type.value for e in session_a.event_engine.event_history}
        types_b = {e.event_type.value for e in session_b.event_engine.event_history}
        self.assertIn("INTRUSION", types_a)
        self.assertIn("INTRUSION", types_b)

    def test_restarting_one_camera_does_not_affect_the_other(self):
        session_b_before, _ = self.cam_b.snapshot()
        frames_before = session_b_before.frame_index

        self.cam_a.request_restart()
        deadline = time.time() + 10
        while time.time() < deadline and self.cam_a.status != "online":
            time.sleep(0.3)

        session_b_after, _ = self.cam_b.snapshot()
        self.assertIs(session_b_before, session_b_after)  # same object, never replaced
        self.assertGreaterEqual(session_b_after.frame_index, frames_before)  # kept advancing, untouched


class TestCameraFailureIsolation(unittest.TestCase):
    """A camera pointed at a nonexistent file must not crash the manager or
    affect a healthy sibling camera — and must be able to recover once its
    configuration is fixed and a restart is requested."""

    @classmethod
    def setUpClass(cls):
        cls.manager = CameraManager()
        cls.good_cam = cls.manager.add_camera(camera_name="Healthy", video_path="data/videos/test.mp4")
        cls.bad_cam = cls.manager.add_camera(camera_name="Broken", video_path="data/videos/does_not_exist.mp4")

    @classmethod
    def tearDownClass(cls):
        cls.manager.shutdown()

    def test_bad_camera_reports_error_not_crash(self):
        deadline = time.time() + 15
        while time.time() < deadline and self.bad_cam.status == "starting":
            time.sleep(0.3)
        self.assertEqual(self.bad_cam.status, "error")
        self.assertIsNotNone(self.bad_cam.error)
        self.assertIn("does_not_exist.mp4", self.bad_cam.error)

    def test_healthy_camera_unaffected_by_sibling_failure(self):
        deadline = time.time() + 15
        while time.time() < deadline and self.good_cam.status == "starting":
            time.sleep(0.3)
        self.assertEqual(self.good_cam.status, "online", self.good_cam.error)

    def test_manager_still_lists_both_cameras(self):
        ids = {c.camera_id for c in self.manager.list_cameras()}
        self.assertEqual(ids, {self.good_cam.camera_id, self.bad_cam.camera_id})

    def test_camera_recovers_after_fixing_path_and_restarting(self):
        self.assertEqual(self.bad_cam.status, "error")  # precondition from the earlier test
        self.bad_cam.config.video_path = "data/videos/test.mp4"  # simulate a corrected source
        self.bad_cam.request_restart()

        deadline = time.time() + 15
        while time.time() < deadline and self.bad_cam.status != "online":
            time.sleep(0.3)
        self.assertEqual(self.bad_cam.status, "online", self.bad_cam.error)


class TestEndOfStreamLooping(unittest.TestCase):
    """A short clip must loop indefinitely rather than leaving the tile dead
    at end-of-stream, and must do so without help from any other camera."""

    @classmethod
    def setUpClass(cls):
        cls.manager = CameraManager()
        cls.cam = cls.manager.add_camera(camera_name="Loop Test", video_path="data/videos/test.mp4")
        deadline = time.time() + 15
        while time.time() < deadline and cls.cam.status == "starting":
            time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.manager.shutdown()

    def test_frame_index_wraps_around_at_least_once(self):
        self.assertEqual(self.cam.status, "online", self.cam.error)
        seen_wrap = False
        last_frame = -1
        deadline = time.time() + 30
        while time.time() < deadline and not seen_wrap:
            session, _ = self.cam.snapshot()
            if session is not None:
                current = session.frame_index
                if current < last_frame:
                    seen_wrap = True
                last_frame = current
            time.sleep(0.5)
        self.assertTrue(seen_wrap, "expected frame_index to reset at least once (video loop)")
        self.assertEqual(self.cam.status, "online")  # still healthy after looping


if __name__ == "__main__":
    unittest.main()
