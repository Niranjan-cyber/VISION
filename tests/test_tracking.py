import unittest
from src.core.types import BoundingBox, Detection, Track
from src.tracking.tracker import ByteTrackTracker


class TestTracking(unittest.TestCase):
    """Lightweight unit tests for VISION tracking domain types and ByteTrack tracker interface."""

    def test_track_dataclass_creation(self):
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        track = Track(
            track_id=1,
            class_id=2,
            class_name="car",
            confidence=0.92,
            bbox=bbox,
            frame_number=10,
        )
        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.class_name, "car")
        self.assertEqual(track.confidence, 0.92)
        self.assertEqual(track.bbox.width, 100)
        self.assertEqual(track.bbox.height, 100)
        self.assertEqual(track.frame_number, 10)

    def test_tracker_initialization_and_empty_update(self):
        tracker = ByteTrackTracker(track_thresh=0.25)
        tracks = tracker.update(detections=[], frame_number=1)
        self.assertIsInstance(tracks, list)
        self.assertEqual(len(tracks), 0)

    def test_tracker_detection_to_track_persistence(self):
        tracker = ByteTrackTracker(track_thresh=0.25)

        # Frame 1
        dets_frame1 = [
            Detection(
                class_id=2,
                class_name="car",
                confidence=0.88,
                bbox=BoundingBox(100, 100, 200, 200),
            )
        ]
        tracks_frame1 = tracker.update(dets_frame1, frame_number=1)
        self.assertEqual(len(tracks_frame1), 1)
        assigned_id = tracks_frame1[0].track_id
        self.assertEqual(tracks_frame1[0].class_name, "car")

        # Frame 2 (consecutive frame with slight movement)
        dets_frame2 = [
            Detection(
                class_id=2,
                class_name="car",
                confidence=0.90,
                bbox=BoundingBox(105, 102, 205, 202),
            )
        ]
        tracks_frame2 = tracker.update(dets_frame2, frame_number=2)
        self.assertEqual(len(tracks_frame2), 1)
        self.assertEqual(tracks_frame2[0].track_id, assigned_id)


if __name__ == "__main__":
    unittest.main()
