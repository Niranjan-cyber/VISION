import unittest
import numpy as np

from src.face.cross_environment_diagnostic import (
    classify_face_quality,
    compute_identity_prototypes,
    calculate_distribution_metrics,
    compute_threshold_acceptance,
)
from src.face.gallery import FaceGallery


class TestCrossEnvironmentDiagnostic(unittest.TestCase):
    """Unit test suite for Slice 5.7: Cross-Environment Face Recognition Diagnostics."""

    def test_1_face_quality_classification(self):
        """TEST 1: Face quality correctly classifies into GOOD, MEDIUM, and POOR."""
        self.assertEqual(classify_face_quality(width=90, height=110, confidence=0.85), "GOOD")
        self.assertEqual(classify_face_quality(width=60, height=70, confidence=0.60), "MEDIUM")
        self.assertEqual(classify_face_quality(width=30, height=40, confidence=0.40), "POOR")
        self.assertEqual(classify_face_quality(width=100, height=100, confidence=0.45), "POOR")  # Low confidence

    def test_2_identity_prototype_computation(self):
        """TEST 2: Prototypes are properly L2-normalized mean vectors."""
        gallery = FaceGallery()
        v1 = np.ones(512, dtype=np.float32) / np.sqrt(512)
        v2 = np.full(512, 0.5, dtype=np.float32) / np.linalg.norm(np.full(512, 0.5, dtype=np.float32))

        gallery.add("TestPerson", v1)
        gallery.add("TestPerson", v2)

        prototypes = compute_identity_prototypes(gallery)
        self.assertIn("TestPerson", prototypes)
        proto = prototypes["TestPerson"]
        self.assertEqual(len(proto), 512)
        norm = np.linalg.norm(proto)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_3_distribution_metrics_calculation(self):
        """TEST 3: Distribution metrics correctly compute percentiles, mean, median, and min/max."""
        sims = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        metrics = calculate_distribution_metrics(sims)

        self.assertEqual(metrics["count"], 9)
        self.assertAlmostEqual(metrics["mean"], 0.50, places=4)
        self.assertAlmostEqual(metrics["median"], 0.50, places=4)
        self.assertAlmostEqual(metrics["min"], 0.10, places=4)
        self.assertAlmostEqual(metrics["max"], 0.90, places=4)
        self.assertAlmostEqual(metrics["p50"], 0.50, places=4)
        self.assertAlmostEqual(metrics["p10"], 0.18, places=2)
        self.assertAlmostEqual(metrics["p90"], 0.82, places=2)

    def test_4_threshold_sweep_acceptance(self):
        """TEST 4: Threshold acceptance calculates correct cumulative percentages."""
        sims = [0.42, 0.53, 0.58, 0.64, 0.71]
        thresholds = [0.40, 0.50, 0.60, 0.70]
        acc = compute_threshold_acceptance(sims, thresholds)

        self.assertAlmostEqual(acc[0.40], 100.0, places=2)  # All 5 >= 0.40
        self.assertAlmostEqual(acc[0.50], 80.0, places=2)   # 4 >= 0.50
        self.assertAlmostEqual(acc[0.60], 40.0, places=2)   # 2 >= 0.60
        self.assertAlmostEqual(acc[0.70], 20.0, places=2)   # 1 >= 0.70

    def test_5_empty_distribution_safe_handling(self):
        """TEST 5: Empty inputs handled gracefully without exceptions."""
        metrics = calculate_distribution_metrics([])
        self.assertEqual(metrics["count"], 0)
        self.assertEqual(metrics["mean"], 0.0)

        acc = compute_threshold_acceptance([], [0.50, 0.60])
        self.assertEqual(acc[0.50], 0.0)


if __name__ == "__main__":
    unittest.main()
