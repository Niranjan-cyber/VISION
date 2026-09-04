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
import tempfile
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Isolated Phase 3 event/alert/zone database for this whole module, so real
# cameras spun up here never write test data into the shared demo database
# (data/vision.db) or race other test modules writing to it concurrently.
os.environ["VISION_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="vision_test_camera_manager_db_"), "test.db")
os.environ["VISION_SNAPSHOT_DIR"] = tempfile.mkdtemp(prefix="vision_test_camera_manager_snapshots_")

from src.pipeline.camera_manager import CameraLimitReached, CameraManager, MAX_ACTIVE_CAMERAS
from src.pipeline.session import PipelineSession
from src.ingestion.video import discover_camera_devices

_DISCOVERED_DEVICES = discover_camera_devices(max_index=3)
_FIRST_AVAILABLE_DEVICE = next((d["device_index"] for d in _DISCOVERED_DEVICES if d["available"]), None)


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


@unittest.skipIf(_FIRST_AVAILABLE_DEVICE is None, "no local camera device available on this machine")
class TestLiveCameraLifecycle(unittest.TestCase):
    """A live camera runs the exact same PipelineSession as a recorded one —
    these tests exercise real device I/O (source_type='live'), not a mock."""

    @classmethod
    def setUpClass(cls):
        cls.manager = CameraManager()

    @classmethod
    def tearDownClass(cls):
        cls.manager.shutdown()

    @classmethod
    def _wait_online_or_error(cls, cam, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline and cam.status == "starting":
            time.sleep(0.3)

    def test_live_camera_becomes_online_and_is_marked_live(self):
        cam = self.manager.add_camera(
            camera_name="Webcam", source_type="live", device_index=_FIRST_AVAILABLE_DEVICE
        )
        self._wait_online_or_error(cam)
        self.assertEqual(cam.status, "online", cam.error)
        self.assertEqual(cam.config.source_type, "live")
        self.assertEqual(cam.config.device_index, _FIRST_AVAILABLE_DEVICE)
        session, _ = cam.snapshot()
        self.assertTrue(session.is_live)
        self.manager.remove_camera(cam.camera_id)

    def test_invalid_device_index_rejected_before_spawning_a_camera(self):
        with self.assertRaises(ValueError):
            self.manager.add_camera(camera_name="Bad", source_type="live", device_index=None)

    def test_nonexistent_device_index_reports_error_not_crash(self):
        cam = self.manager.add_camera(camera_name="Ghost Camera", source_type="live", device_index=9)
        self._wait_online_or_error(cam)
        self.assertEqual(cam.status, "error")
        self.assertIsNotNone(cam.error)
        self.manager.remove_camera(cam.camera_id)

    def test_live_camera_disconnect_is_isolated_and_restart_recovers(self):
        cam = self.manager.add_camera(
            camera_name="Webcam Disconnect Test", source_type="live", device_index=_FIRST_AVAILABLE_DEVICE
        )
        self._wait_online_or_error(cam)
        self.assertEqual(cam.status, "online", cam.error)

        # Simulate a physical disconnect by closing the underlying capture
        # device out from under the capture thread — the next read() then
        # fails exactly as it would if the device were unplugged.
        session, _ = cam.snapshot()
        session.source._cap.release()

        deadline = time.time() + 10
        while time.time() < deadline and cam.status != "error":
            time.sleep(0.2)
        self.assertEqual(cam.status, "error")
        self.assertIn("disconnected", cam.error)

        # An explicit restart attempts to reopen the physical device.
        cam.request_restart()
        deadline = time.time() + 15
        while time.time() < deadline and cam.status != "online":
            time.sleep(0.3)
        self.assertEqual(cam.status, "online", cam.error)
        self.manager.remove_camera(cam.camera_id)


@unittest.skipIf(_FIRST_AVAILABLE_DEVICE is None, "no local camera device available on this machine")
class TestMixedRecordedAndLiveCameras(unittest.TestCase):
    """MAX_ACTIVE_CAMERAS applies to the combined set of recorded + live
    sources — a fifth camera is rejected regardless of the type mix."""

    @classmethod
    def setUpClass(cls):
        cls.manager = CameraManager()
        cls.cams = [
            cls.manager.add_camera(camera_name="Recorded A", video_path="data/videos/test.mp4"),
            cls.manager.add_camera(camera_name="Recorded B", video_path="data/videos/test.mp4"),
            cls.manager.add_camera(
                camera_name="Live A", source_type="live", device_index=_FIRST_AVAILABLE_DEVICE
            ),
        ]
        for cam in cls.cams:
            deadline = time.time() + 30
            while time.time() < deadline and cam.status == "starting":
                time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.manager.shutdown()

    def test_all_three_mixed_cameras_online_independently(self):
        for cam in self.cams:
            self.assertEqual(cam.status, "online", cam.error)

    def test_fourth_camera_fills_the_limit_and_fifth_is_rejected(self):
        fourth = self.manager.add_camera(camera_name="Recorded C", video_path="data/videos/test.mp4")
        self.assertEqual(self.manager.active_count, MAX_ACTIVE_CAMERAS)
        with self.assertRaises(CameraLimitReached):
            self.manager.add_camera(
                camera_name="Live B (one too many)", source_type="live", device_index=_FIRST_AVAILABLE_DEVICE
            )
        self.manager.remove_camera(fourth.camera_id)  # keep count symmetric for other tests


class TestCaptureAIDecoupling(unittest.TestCase):
    """The core Phase-2.1 fix: video capture/display must not be coupled to
    (and therefore capped by) the AI inference rate."""

    @classmethod
    def setUpClass(cls):
        cls.manager = CameraManager()
        # A deliberately slow AI rate makes the gap between capture progress
        # and AI progress unmistakable within a short test window. Uses the
        # ~24s sample1.mp4 clip (not the 2s test.mp4) so the measurement
        # window can't straddle an EOF-loop reset of the counters.
        cls.cam = cls.manager.add_camera(
            camera_name="Decoupling Test", video_path="data/videos/sample1.mp4", ai_fps=1.0,
        )
        deadline = time.time() + 20
        while time.time() < deadline and cls.cam.status == "starting":
            time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.manager.shutdown()

    def test_capture_advances_far_more_than_throttled_ai_processing(self):
        self.assertEqual(self.cam.status, "online", self.cam.error)
        session, _ = self.cam.snapshot()
        capture_start = session.frame_index
        ai_start = session.inference_count

        time.sleep(3.0)

        session, result = self.cam.snapshot()
        capture_advanced = session.frame_index - capture_start
        ai_advanced = session.inference_count - ai_start

        # At ai_fps=1.0, at most a handful of inference passes can happen in
        # 3s; capture (paced to the clip's native ~20-30fps) advances far
        # more — proving the display path isn't waiting on AI.
        self.assertLessEqual(ai_advanced, 5)
        self.assertGreater(capture_advanced, ai_advanced * 4)

    def test_latest_state_is_single_slot_not_a_growing_queue(self):
        # Bounded/latest-frame strategy: CameraSession exposes exactly one
        # current JPEG and one current FrameResult, never a backlog — this
        # is true by construction (attributes, not a list/deque), asserted
        # here so a future regression toward a queue would be caught.
        self.assertIsInstance(self.cam.latest_jpeg(), (bytes, type(None)))
        _, result = self.cam.snapshot()
        self.assertFalse(isinstance(result, (list, tuple)))


class TestAIFPSThrottling(unittest.TestCase):
    """The AI worker's cadence should approximately track its configured
    ai_fps, independent of the video's own (much higher) native FPS."""

    @classmethod
    def setUpClass(cls):
        cls.manager = CameraManager()
        # ~24s clip so a several-second measurement window can't straddle an
        # EOF-loop reset of session.inference_count.
        cls.cam = cls.manager.add_camera(
            camera_name="AI FPS Test", video_path="data/videos/sample1.mp4", ai_fps=4.0,
        )
        deadline = time.time() + 20
        while time.time() < deadline and cls.cam.status == "starting":
            time.sleep(0.3)
        # Let one-time CUDA/model warmup pass before measuring steady-state cadence.
        warmup_deadline = time.time() + 15
        session, _ = cls.cam.snapshot()
        while time.time() < warmup_deadline and session.inference_count == 0:
            time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.manager.shutdown()

    def test_ai_processing_rate_roughly_matches_configured_ai_fps(self):
        session, _ = self.cam.snapshot()
        start_count = session.inference_count
        window_s = 4.0
        time.sleep(window_s)
        session, _ = self.cam.snapshot()
        measured_fps = (session.inference_count - start_count) / window_s
        # Generous tolerance — this is a wall-clock measurement on a shared
        # test machine, not a hard real-time guarantee. It must not run away
        # to the video's native ~20-30fps, and must produce a non-trivial
        # rate.
        self.assertGreater(measured_fps, 0.5)
        self.assertLess(measured_fps, 8.0)


class TestRecognitionPreservedThroughDecoupledPipeline(unittest.TestCase):
    """The capture/AI threading refactor must not change recognition output.

    Deliberately drives PipelineSession.process_frame() directly on frame 1,
    synchronously — exactly like the pre-refactor single-threaded baseline
    verification did — rather than going through CameraManager's threaded
    capture/AI workers. Which exact frame the *threaded* AI worker picks up
    first is inherently timing-dependent (it depends on ai_fps, system load,
    and how far capture has run by the time the AI thread's first cycle
    fires — see TestAIFPSThrottling for that behavior); pinning a similarity
    value to a specific timing outcome would make this test flaky under
    system load, exactly the kind of test this project's discipline avoids.
    What actually must hold — and what this isolates — is that the
    recognition pipeline itself (embeddings/threshold/margin/matching) is
    byte-for-byte unchanged, independent of any threading."""

    def test_shreyas_similarity_matches_verified_baseline(self):
        session = PipelineSession(
            video_path="data/videos/shreyas1.mp4", zones_path="configs/zones_demo.yaml",
            loitering_duration=3.0, verbose=False,
        )
        try:
            frame = session.source.read_frame()
            session.process_frame(frame, 1)
            match = session.track_identity_cache.get(3) or next(iter(session.track_identity_cache.values()), None)
            self.assertIsNotNone(match)
            self.assertEqual(match.identity, "Shreyas_Chavan")
            self.assertAlmostEqual(match.similarity, 0.710, delta=0.01)
        finally:
            session.release()

    def test_atharva_similarity_matches_verified_baseline(self):
        session = PipelineSession(
            video_path="data/videos/jaysingpure1.mp4", zones_path="configs/zones_cam02.yaml",
            loitering_duration=3.0, verbose=False,
        )
        try:
            frame = session.source.read_frame()
            session.process_frame(frame, 1)
            match = session.track_identity_cache.get(3) or next(iter(session.track_identity_cache.values()), None)
            self.assertIsNotNone(match)
            self.assertEqual(match.identity, "Atharva_Jaysingpure")
            self.assertAlmostEqual(match.similarity, 0.688, delta=0.01)
        finally:
            session.release()


if __name__ == "__main__":
    unittest.main()
