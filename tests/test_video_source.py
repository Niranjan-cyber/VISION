"""
Unit tests for the VideoSource ingestion abstraction (recorded file vs live
local camera device) and camera device discovery. No AI models are loaded
here — these are fast, pure-ingestion tests; CameraManager-level integration
(PipelineSession + threading) is covered separately in test_camera_manager.py.

Live-device tests are skipped (not failed) when this machine has no real
camera attached — asserting hardware that doesn't exist would be exactly the
kind of fabrication this project's testing culture avoids.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.video import VideoSource, discover_camera_devices

_DISCOVERED = discover_camera_devices(max_index=3)
_FIRST_AVAILABLE_DEVICE = next((d["device_index"] for d in _DISCOVERED if d["available"]), None)
_UNLIKELY_DEVICE_INDEX = 9  # far past any real device on a normal machine, cheap enough to probe


class TestVideoSourceFromFile(unittest.TestCase):
    def test_opens_existing_recorded_file(self):
        src = VideoSource(video_path="data/videos/test.mp4")
        try:
            self.assertFalse(src.is_live)
            self.assertGreater(src.width, 0)
            self.assertGreater(src.height, 0)
            self.assertGreater(src.fps, 0)
        finally:
            src.release()

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            VideoSource(video_path="data/videos/does_not_exist_at_all.mp4")

    def test_read_frame_returns_none_at_eof(self):
        src = VideoSource(video_path="data/videos/test.mp4")
        try:
            frame = src.read_frame()
            while frame is not None:
                frame = src.read_frame()
            self.assertIsNone(src.read_frame())  # stays None, doesn't raise
        finally:
            src.release()


class TestVideoSourceArgumentValidation(unittest.TestCase):
    def test_neither_path_nor_device_index_raises(self):
        with self.assertRaises(ValueError):
            VideoSource()

    def test_both_path_and_device_index_raises(self):
        with self.assertRaises(ValueError):
            VideoSource(video_path="data/videos/test.mp4", device_index=0)


class TestInvalidCameraDevice(unittest.TestCase):
    def test_nonexistent_device_index_raises_value_error_not_crash(self):
        with self.assertRaises(ValueError):
            VideoSource(device_index=_UNLIKELY_DEVICE_INDEX)


@unittest.skipIf(_FIRST_AVAILABLE_DEVICE is None, "no local camera device available on this machine")
class TestVideoSourceFromLiveDevice(unittest.TestCase):
    def test_opens_real_local_device(self):
        src = VideoSource(device_index=_FIRST_AVAILABLE_DEVICE)
        try:
            self.assertTrue(src.is_live)
            frame = src.read_frame()
            self.assertIsNotNone(frame)
        finally:
            src.release()

    def test_frame_count_is_unbounded_for_live_source(self):
        src = VideoSource(device_index=_FIRST_AVAILABLE_DEVICE)
        try:
            self.assertEqual(src.frame_count, 0)  # unknown/unbounded, never guessed
        finally:
            src.release()


class TestDiscoverCameraDevices(unittest.TestCase):
    def test_returns_expected_shape_for_every_probed_index(self):
        results = discover_camera_devices(max_index=3)
        self.assertEqual(len(results), 3)
        for i, entry in enumerate(results):
            self.assertEqual(entry["device_index"], i)
            self.assertIn("available", entry)
            self.assertIsInstance(entry["available"], bool)
            self.assertIn("width", entry)
            self.assertIn("height", entry)

    def test_never_claims_a_device_is_available_without_reading_a_frame(self):
        # An index far beyond any real device must always come back
        # unavailable — discovery must never invent a camera.
        results = discover_camera_devices(max_index=_UNLIKELY_DEVICE_INDEX + 1)
        self.assertFalse(results[_UNLIKELY_DEVICE_INDEX]["available"])
        self.assertEqual(results[_UNLIKELY_DEVICE_INDEX]["width"], 0)


if __name__ == "__main__":
    unittest.main()
