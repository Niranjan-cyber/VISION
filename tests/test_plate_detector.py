import unittest
import cv2
import numpy as np
from src.anpr.detector import LicensePlateDetector


class TestPlateDetector(unittest.TestCase):
    def setUp(self):
        self.detector = LicensePlateDetector()

    def test_invalid_crops_return_empty(self):
        self.assertEqual(self.detector.detect(None), [])
        self.assertEqual(self.detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)), [])
        self.assertEqual(self.detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)), [])

    def test_morphological_detection_on_synthetic_plate(self):
        # Create a mock vehicle crop: 300x400 dark background
        vehicle = np.full((300, 400, 3), 40, dtype=np.uint8)

        # Draw a white rectangular license plate in lower half:
        # width=140, height=40 (aspect ratio = 3.5)
        px1, py1, px2, py2 = 130, 200, 270, 240
        cv2.rectangle(vehicle, (px1, py1), (px2, py2), (255, 255, 255), -1)

        # Draw simulated dark characters with vertical strokes inside the plate
        for x in range(px1 + 10, px2 - 10, 12):
            cv2.line(vehicle, (x, py1 + 8), (x, py2 - 8), (0, 0, 0), 3)

        detections = self.detector.detect(vehicle)
        self.assertGreater(len(detections), 0)

        best = detections[0]
        # Bounding box should overlap closely with our mock plate
        self.assertGreater(best.confidence, 0.40)
        self.assertLess(abs(best.bbox.x1 - px1), 30)
        self.assertLess(abs(best.bbox.y1 - py1), 30)


if __name__ == "__main__":
    unittest.main()
