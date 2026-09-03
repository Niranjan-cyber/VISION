import unittest
import cv2
import numpy as np

from src.core.types import BoundingBox, PlateRecognitionResult, Track
from src.anpr.association import associate_plates_to_vehicles, map_crop_to_global_bbox
from src.anpr.cache import PlateTrackCache
from src.anpr.cleaner import clean_plate_text
from src.anpr.detector import LicensePlateDetector
from src.anpr.enhancer import PlateEnhancer
from src.anpr.ocr import MockPlateOCREngine


class TestANPRPipelineIntegration(unittest.TestCase):
    def test_full_anpr_pipeline(self):
        # 1. Setup simulated frame with vehicle and plate
        frame_w, frame_h = 1280, 720
        frame = np.full((frame_h, frame_w, 3), 50, dtype=np.uint8)

        # Vehicle bbox: [200, 150, 600, 450] (car)
        vx1, vy1, vx2, vy2 = 200, 150, 600, 450
        vehicle_crop = frame[vy1:vy2, vx1:vx2]

        # Draw a synthetic white plate on vehicle
        px1, py1, px2, py2 = 120, 200, 260, 240
        cv2.rectangle(vehicle_crop, (px1, py1), (px2, py2), (255, 255, 255), -1)
        for x in range(px1 + 10, px2 - 10, 14):
            cv2.line(vehicle_crop, (x, py1 + 6), (x, py2 - 6), (0, 0, 0), 2)

        # 2. License Plate Detection on vehicle crop
        detector = LicensePlateDetector()
        plate_detections = detector.detect(vehicle_crop)
        self.assertGreater(len(plate_detections), 0)
        best_plate_det = plate_detections[0]

        # 3. Plate Crop & Enhancement
        pb = best_plate_det.bbox
        plate_crop = vehicle_crop[pb.y1:pb.y2, pb.x1:pb.x2]
        enhancer = PlateEnhancer(target_height=70)
        enhanced_crop = enhancer.enhance(plate_crop)
        self.assertIsNotNone(enhanced_crop)

        # 4. OCR
        ocr_engine = MockPlateOCREngine(fixed_text="MH-12-AB-1234", confidence=0.94)
        raw_text, ocr_conf = ocr_engine.recognize(enhanced_crop)

        # 5. Text Cleaning & Syntax Validation
        cleaned_text, is_valid, mult = clean_plate_text(raw_text)
        self.assertEqual(cleaned_text, "MH12AB1234")
        self.assertTrue(is_valid)
        final_conf = ocr_conf * mult

        # 6. Global Coordinate Mapping
        global_plate_bbox = map_crop_to_global_bbox(
            pb,
            BoundingBox(x1=vx1, y1=vy1, x2=vx2, y2=vy2),
            frame_w,
            frame_h,
        )

        plate_result = PlateRecognitionResult(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            confidence=final_conf,
            is_valid=is_valid,
            bbox=global_plate_bbox,
        )

        # 7. Association to Vehicle Track
        car_track = Track(
            track_id=42,
            class_id=2,
            class_name="car",
            confidence=0.88,
            bbox=BoundingBox(x1=vx1, y1=vy1, x2=vx2, y2=vy2),
            frame_number=1,
        )
        associations = associate_plates_to_vehicles([car_track], [plate_result])
        self.assertEqual(len(associations), 1)
        self.assertEqual(associations[0].track_id, 42)
        self.assertEqual(associations[0].plate.cleaned_text, "MH12AB1234")

        # 8. Track-Level Caching
        cache = PlateTrackCache()
        cached_result = cache.update(42, plate_result, frame_number=1)
        self.assertEqual(cached_result.cleaned_text, "MH12AB1234")
        self.assertTrue(cached_result.is_valid)


if __name__ == "__main__":
    unittest.main()
