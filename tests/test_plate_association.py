import unittest
from src.core.types import BoundingBox, PlateRecognitionResult, Track
from src.anpr.association import associate_plates_to_vehicles, map_crop_to_global_bbox


class TestPlateAssociation(unittest.TestCase):
    def test_map_crop_to_global_bbox(self):
        parent_bbox = BoundingBox(x1=100, y1=200, x2=500, y2=600)
        crop_bbox = BoundingBox(x1=20, y1=50, x2=160, y2=90)

        global_bbox = map_crop_to_global_bbox(crop_bbox, parent_bbox, frame_w=1920, frame_h=1080)
        self.assertEqual(global_bbox.x1, 120)
        self.assertEqual(global_bbox.y1, 250)
        self.assertEqual(global_bbox.x2, 260)
        self.assertEqual(global_bbox.y2, 290)

    def test_associate_plates_to_vehicles(self):
        # Car track 12
        car_track = Track(
            track_id=12,
            class_id=2,
            class_name="car",
            confidence=0.90,
            bbox=BoundingBox(x1=100, y1=200, x2=500, y2=600),
            frame_number=1,
        )

        # Person track 3 (should be ignored by vehicle association)
        person_track = Track(
            track_id=3,
            class_id=0,
            class_name="person",
            confidence=0.88,
            bbox=BoundingBox(x1=600, y1=200, x2=700, y2=600),
            frame_number=1,
        )

        # Plate located inside car track
        plate = PlateRecognitionResult(
            raw_text="MH12AB1234",
            cleaned_text="MH12AB1234",
            confidence=0.95,
            is_valid=True,
            bbox=BoundingBox(x1=200, y1=450, x2=350, y2=500),
        )

        associations = associate_plates_to_vehicles([car_track, person_track], [plate])
        self.assertEqual(len(associations), 1)
        self.assertEqual(associations[0].track_id, 12)
        self.assertEqual(associations[0].vehicle_class, "car")
        self.assertEqual(associations[0].plate.cleaned_text, "MH12AB1234")


if __name__ == "__main__":
    unittest.main()
