import unittest
import numpy as np

from src.core.types import BoundingBox, FaceDetection, Track
from src.face.association import FaceTrackAssociation, associate_faces_to_tracks
from src.face.detector import FaceDetector


class TestFaceModule(unittest.TestCase):
    """Unit test suite for Vertical Slice 3: Face Detection & Person-Track Association."""

    def test_1_face_detection_dataclass_creation(self):
        """TEST 1: FaceDetection dataclass can be created."""
        bbox = BoundingBox(x1=10, y1=20, x2=50, y2=60)
        face = FaceDetection(bbox=bbox, confidence=0.88)
        self.assertEqual(face.confidence, 0.88)
        self.assertEqual(face.bbox.x1, 10)
        self.assertEqual(face.bbox.y1, 20)
        self.assertEqual(face.bbox.x2, 50)
        self.assertEqual(face.bbox.y2, 60)

    def test_2_face_center_inside_person_bbox(self):
        """TEST 2: Face center inside person bbox -> association succeeds."""
        person_track = Track(
            track_id=17,
            class_id=0,
            class_name="person",
            confidence=0.94,
            bbox=BoundingBox(x1=100, y1=50, x2=300, y2=450),
            frame_number=1,
        )
        # Face at (160, 80) -> center is (180, 110), inside person (100..300, 50..450)
        face = FaceDetection(
            bbox=BoundingBox(x1=160, y1=80, x2=200, y2=140), confidence=0.90
        )

        associations = associate_faces_to_tracks([person_track], [face])
        self.assertEqual(len(associations), 1)
        self.assertEqual(associations[0].track_id, 17)
        self.assertEqual(associations[0].face, face)

    def test_3_face_center_outside_person_bbox(self):
        """TEST 3: Face center outside person bbox -> no association."""
        person_track = Track(
            track_id=5,
            class_id=0,
            class_name="person",
            confidence=0.90,
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=300),
            frame_number=1,
        )
        # Face at (500, 500) -> far outside person box
        face = FaceDetection(
            bbox=BoundingBox(x1=500, y1=500, x2=550, y2=550), confidence=0.85
        )

        associations = associate_faces_to_tracks([person_track], [face])
        self.assertEqual(len(associations), 0)

    def test_4_car_track_cannot_receive_face_association(self):
        """TEST 4: A car Track cannot receive a face association."""
        car_track = Track(
            track_id=7,
            class_id=2,
            class_name="car",
            confidence=0.91,
            bbox=BoundingBox(x1=100, y1=100, x2=300, y2=300),
            frame_number=1,
        )
        # Face inside car bbox boundaries
        face = FaceDetection(
            bbox=BoundingBox(x1=150, y1=150, x2=200, y2=200), confidence=0.89
        )

        associations = associate_faces_to_tracks([car_track], [face])
        self.assertEqual(len(associations), 0)

    def test_5_multiple_person_tracks_receive_correct_faces(self):
        """TEST 5: Multiple person tracks receive the correct faces."""
        person1 = Track(
            track_id=10,
            class_id=0,
            class_name="person",
            confidence=0.92,
            bbox=BoundingBox(x1=50, y1=50, x2=150, y2=250),
            frame_number=1,
        )
        person2 = Track(
            track_id=20,
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=BoundingBox(x1=400, y1=100, x2=550, y2=400),
            frame_number=1,
        )

        face1 = FaceDetection(
            bbox=BoundingBox(x1=80, y1=60, x2=120, y2=100), confidence=0.88
        )
        face2 = FaceDetection(
            bbox=BoundingBox(x1=440, y1=120, x2=500, y2=180), confidence=0.91
        )

        associations = associate_faces_to_tracks([person1, person2], [face1, face2])
        self.assertEqual(len(associations), 2)

        assoc_map = {assoc.track_id: assoc.face for assoc in associations}
        self.assertIn(10, assoc_map)
        self.assertIn(20, assoc_map)
        self.assertEqual(assoc_map[10], face1)
        self.assertEqual(assoc_map[20], face2)

    def test_6_crop_relative_coordinates_transformed_to_full_frame(self):
        """TEST 6: Crop-relative face coordinates are correctly transformed into full-frame coordinates."""
        person_x1, person_y1 = 100, 50

        # Face detector output relative to crop
        crop_face = FaceDetection(
            bbox=BoundingBox(x1=60, y1=30, x2=140, y2=120), confidence=0.87
        )

        # Transformation to full-frame global coordinates
        global_x1 = person_x1 + crop_face.bbox.x1
        global_y1 = person_y1 + crop_face.bbox.y1
        global_x2 = person_x1 + crop_face.bbox.x2
        global_y2 = person_y1 + crop_face.bbox.y2

        global_face_bbox = BoundingBox(
            x1=global_x1, y1=global_y1, x2=global_x2, y2=global_y2
        )

        self.assertEqual(global_face_bbox.x1, 160)
        self.assertEqual(global_face_bbox.y1, 80)
        self.assertEqual(global_face_bbox.x2, 240)
        self.assertEqual(global_face_bbox.y2, 170)

    def test_7_invalid_empty_person_crop_handled_safely(self):
        """TEST 7: Invalid/empty person crop is handled safely."""
        detector = FaceDetector(score_threshold=0.5)

        # Test empty crop
        empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
        faces_empty = detector.detect(empty_crop)
        self.assertIsInstance(faces_empty, list)
        self.assertEqual(len(faces_empty), 0)

        # Test None crop
        faces_none = detector.detect(None)
        self.assertIsInstance(faces_none, list)
        self.assertEqual(len(faces_none), 0)

        # Test tiny crop < 5px
        tiny_crop = np.zeros((2, 2, 3), dtype=np.uint8)
        faces_tiny = detector.detect(tiny_crop)
        self.assertIsInstance(faces_tiny, list)
        self.assertEqual(len(faces_tiny), 0)


if __name__ == "__main__":
    unittest.main()
