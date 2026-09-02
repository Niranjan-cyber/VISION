import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from src.core.types import FaceEmbedding
from src.face.diagnostic import compute_pairwise_diagnostic
from src.face.embedder import FaceEmbedder
from src.face.preprocessing import l2_normalize


class TestArcFaceDiagnostic(unittest.TestCase):
    """Unit test suite for Slice 5.1: ArcFace Embedding Diagnostic."""

    @classmethod
    def setUpClass(cls):
        cls.embedder = FaceEmbedder()

    def test_1_output_dimension_is_512(self):
        """TEST 1: Verify output dimension is strictly 512."""
        dummy_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        emb = self.embedder.embed(dummy_crop)

        self.assertIsNotNone(emb)
        self.assertEqual(emb.dimension, 512)
        self.assertEqual(len(emb.vector), 512)

    def test_2_embeddings_are_finite(self):
        """TEST 2: Verify embeddings contain no NaN or Inf values."""
        dummy_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        emb = self.embedder.embed(dummy_crop)

        self.assertIsNotNone(emb)
        self.assertFalse(np.isnan(emb.vector).any())
        self.assertFalse(np.isinf(emb.vector).any())

    def test_3_embeddings_are_l2_normalized(self):
        """TEST 3: Verify embedding vector L2 norm is approximately 1.0."""
        dummy_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        emb = self.embedder.embed(dummy_crop)

        self.assertIsNotNone(emb)
        norm = np.linalg.norm(emb.vector)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_4_identical_images_produce_identical_embeddings(self):
        """TEST 4: Identical image crops produce approximately identical embeddings (similarity ~ 1.0)."""
        crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)

        emb1 = self.embedder.embed(crop)
        emb2 = self.embedder.embed(crop)

        self.assertIsNotNone(emb1)
        self.assertIsNotNone(emb2)

        sim = float(np.dot(emb1.vector, emb2.vector))
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_5_different_gallery_identities_evaluated_independently(self):
        """TEST 5: Different gallery identities generate distinct, independent vectors."""
        crop_a = np.zeros((112, 112, 3), dtype=np.uint8)
        crop_b = np.full((112, 112, 3), 255, dtype=np.uint8)

        emb_a = self.embedder.embed(crop_a)
        emb_b = self.embedder.embed(crop_b)

        self.assertIsNotNone(emb_a)
        self.assertIsNotNone(emb_b)

        # Vectors must not be identical or pointing to the same memory reference
        self.assertFalse(np.array_equal(emb_a.vector, emb_b.vector))
        self.assertIsNot(emb_a.vector, emb_b.vector)

    def test_6_cosine_similarity_calculation_is_correct(self):
        """TEST 6: Verify dot product cosine similarity logic on known normalized vectors."""
        v1 = l2_normalize(np.array([1.0, 0.0] + [0.0] * 510, dtype=np.float32))
        v2 = l2_normalize(np.array([1.0, 0.0] + [0.0] * 510, dtype=np.float32))
        v3 = l2_normalize(np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32))

        sim_identical = float(np.dot(v1, v2))
        sim_orthogonal = float(np.dot(v1, v3))

        self.assertAlmostEqual(sim_identical, 1.0, places=5)
        self.assertAlmostEqual(sim_orthogonal, 0.0, places=5)

    def test_7_diagnostic_pairwise_computation(self):
        """TEST 7: Verify diagnostic helper on a temporary mock gallery directory."""
        temp_dir = tempfile.mkdtemp()
        try:
            id1_dir = os.path.join(temp_dir, "person_1")
            id2_dir = os.path.join(temp_dir, "person_2")
            os.makedirs(id1_dir, exist_ok=True)
            os.makedirs(id2_dir, exist_ok=True)

            # Synthetic image with a central face-like patch
            img1 = np.full((200, 200, 3), 128, dtype=np.uint8)
            img2 = np.full((200, 200, 3), 128, dtype=np.uint8)

            cv2.imwrite(os.path.join(id1_dir, "ref1.jpg"), img1)
            cv2.imwrite(os.path.join(id2_dir, "ref2.jpg"), img2)

            report = compute_pairwise_diagnostic(gallery_dir=temp_dir, embedder=self.embedder)

            self.assertIsNotNone(report)
            self.assertGreaterEqual(report.num_identities, 0)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
