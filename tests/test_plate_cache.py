import unittest
from src.core.types import BoundingBox, PlateRecognitionResult
from src.anpr.cache import PlateTrackCache


class TestPlateCache(unittest.TestCase):
    def setUp(self):
        self.cache = PlateTrackCache(confidence_boost_threshold=0.05)

    def test_initial_cache_and_retrieval(self):
        plate1 = PlateRecognitionResult(
            raw_text="MH12AB1234",
            cleaned_text="MH12AB1234",
            confidence=0.75,
            is_valid=True,
            bbox=BoundingBox(10, 10, 50, 30),
        )

        res = self.cache.update(track_id=12, plate=plate1, frame_number=1)
        self.assertEqual(res.cleaned_text, "MH12AB1234")

        cached = self.cache.get(12)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.cleaned_text, "MH12AB1234")
        self.assertEqual(cached.confidence, 0.75)

    def test_higher_confidence_updates_record(self):
        plate_low = PlateRecognitionResult(
            raw_text="MH12AB1234",
            cleaned_text="MH12AB1234",
            confidence=0.70,
            is_valid=True,
            bbox=BoundingBox(10, 10, 50, 30),
        )
        self.cache.update(track_id=12, plate=plate_low, frame_number=1)

        plate_high = PlateRecognitionResult(
            raw_text="MH12AB1234",
            cleaned_text="MH12AB1234",
            confidence=0.92,
            is_valid=True,
            bbox=BoundingBox(12, 12, 52, 32),
        )
        self.cache.update(track_id=12, plate=plate_high, frame_number=2)

        cached = self.cache.get(12)
        self.assertAlmostEqual(cached.confidence, 0.92)

    def test_valid_plate_overrides_invalid_plate(self):
        # Frame 1: poor OCR produces invalid plate
        invalid_plate = PlateRecognitionResult(
            raw_text="MH12",
            cleaned_text="MH12",
            confidence=0.85,
            is_valid=False,
            bbox=BoundingBox(10, 10, 50, 30),
        )
        self.cache.update(track_id=15, plate=invalid_plate, frame_number=1)

        # Frame 2: valid syntax plate observed
        valid_plate = PlateRecognitionResult(
            raw_text="MH12AB1234",
            cleaned_text="MH12AB1234",
            confidence=0.75,
            is_valid=True,
            bbox=BoundingBox(10, 10, 50, 30),
        )
        self.cache.update(track_id=15, plate=valid_plate, frame_number=2)

        cached = self.cache.get(15)
        self.assertEqual(cached.cleaned_text, "MH12AB1234")
        self.assertTrue(cached.is_valid)


if __name__ == "__main__":
    unittest.main()
