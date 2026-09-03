import unittest
import numpy as np
from src.anpr.enhancer import PlateEnhancer, enhance_plate_crop


class TestPlateEnhancer(unittest.TestCase):
    def setUp(self):
        self.enhancer = PlateEnhancer(target_height=70)

    def test_invalid_and_empty_crops(self):
        self.assertIsNone(self.enhancer.enhance(None))
        self.assertIsNone(self.enhancer.enhance(np.zeros((0, 0, 3), dtype=np.uint8)))
        self.assertIsNone(self.enhancer.enhance(np.zeros((5, 5, 3), dtype=np.uint8)))

    def test_valid_bgr_crop_enhancement(self):
        # Create a mock plate crop (30x120 BGR)
        dummy_plate = np.random.randint(50, 200, (30, 120, 3), dtype=np.uint8)
        enhanced = self.enhancer.enhance(dummy_plate)

        self.assertIsNotNone(enhanced)
        self.assertEqual(len(enhanced.shape), 2)  # Single channel (grayscale/contrast enhanced)
        self.assertEqual(enhanced.shape[0], 70)   # Scaled to target height
        self.assertTrue(enhanced.dtype == np.uint8)

    def test_convenience_function(self):
        dummy_plate = np.full((40, 140, 3), 128, dtype=np.uint8)
        enhanced = enhance_plate_crop(dummy_plate)
        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape[0], 70)


if __name__ == "__main__":
    unittest.main()
