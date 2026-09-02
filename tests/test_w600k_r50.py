import os
import unittest
import numpy as np

from src.face.modern_embedder import W600KR50Embedder
from src.face.w600k_preprocessing import preprocess_w600k_crop


class TestW600KR50Model(unittest.TestCase):
    """Unit test suite for Slice 5.5: Modern InsightFace w600k_r50 recognition model."""

    @classmethod
    def setUpClass(cls):
        cls.model_path = "models/w600k_r50.onnx"
        if os.path.exists(cls.model_path):
            cls.embedder = W600KR50Embedder(cls.model_path)
        else:
            cls.embedder = None

    def test_1_preprocessing_shape_and_dtype(self):
        """TEST 1: Preprocessing produces (1, 3, 112, 112) float32 tensor."""
        dummy_crop = np.random.randint(0, 256, (90, 80, 3), dtype=np.uint8)
        tensor = preprocess_w600k_crop(dummy_crop, target_size=(112, 112), is_bgr=True)

        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (1, 3, 112, 112))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.min(tensor) >= -1.01)
        self.assertTrue(np.max(tensor) <= 1.01)

    def test_2_model_exists_and_loads(self):
        """TEST 2: Model file exists and loads session correctly."""
        if self.embedder is None:
            self.skipTest(f"Model '{self.model_path}' not yet present.")
        self.assertIsNotNone(self.embedder.session)
        self.assertIsNotNone(self.embedder.input_name)
        self.assertIsNotNone(self.embedder.output_name)

    def test_3_input_and_output_metadata(self):
        """TEST 3: Model input and output metadata match expected contracts."""
        if self.embedder is None:
            self.skipTest("Embedder not loaded.")
        # Typically shape [1, 3, 112, 112] or dynamic batch ['None'/'unk__', 3, 112, 112]
        self.assertEqual(len(self.embedder.input_shape), 4)
        self.assertEqual(self.embedder.input_shape[1:], [3, 112, 112])
        self.assertEqual(self.embedder.embedding_dim, 512)

    def test_4_embedding_dimension_and_norm(self):
        """TEST 4: Embedding vector has 512 dimensions and L2 norm of 1.0."""
        if self.embedder is None:
            self.skipTest("Embedder not loaded.")
        dummy_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        emb = self.embedder.embed(dummy_crop)

        self.assertIsNotNone(emb)
        self.assertEqual(emb.dimension, 512)
        self.assertEqual(len(emb.vector), 512)
        self.assertEqual(emb.vector.dtype, np.float32)
        norm = np.linalg.norm(emb.vector)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_5_embedding_is_finite(self):
        """TEST 5: Embedding values are finite (no NaNs or Infs)."""
        if self.embedder is None:
            self.skipTest("Embedder not loaded.")
        dummy_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        emb = self.embedder.embed(dummy_crop)

        self.assertIsNotNone(emb)
        self.assertFalse(np.isnan(emb.vector).any())
        self.assertFalse(np.isinf(emb.vector).any())

    def test_6_deterministic_inference(self):
        """TEST 6: Repeated inference on the same input produces identical output."""
        if self.embedder is None:
            self.skipTest("Embedder not loaded.")
        dummy_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        emb1 = self.embedder.embed(dummy_crop)
        emb2 = self.embedder.embed(dummy_crop)

        self.assertIsNotNone(emb1)
        self.assertIsNotNone(emb2)
        self.assertTrue(np.allclose(emb1.vector, emb2.vector, atol=1e-5))

    def test_7_different_images_produce_different_embeddings(self):
        """TEST 7: Different inputs produce distinct embeddings."""
        if self.embedder is None:
            self.skipTest("Embedder not loaded.")
        crop1 = np.zeros((112, 112, 3), dtype=np.uint8)
        crop2 = np.full((112, 112, 3), 255, dtype=np.uint8)

        emb1 = self.embedder.embed(crop1)
        emb2 = self.embedder.embed(crop2)

        self.assertIsNotNone(emb1)
        self.assertIsNotNone(emb2)
        self.assertFalse(np.array_equal(emb1.vector, emb2.vector))

    def test_8_random_images_do_not_collapse(self):
        """TEST 8: Random inputs do not collapse into ~0.995 cosine similarity."""
        if self.embedder is None:
            self.skipTest("Embedder not loaded.")
        np.random.seed(42)
        rand_embs = [
            self.embedder.embed(np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)).vector
            for _ in range(5)
        ]
        sims = []
        for i in range(5):
            for j in range(i + 1, 5):
                sims.append(float(np.dot(rand_embs[i], rand_embs[j])))

        mean_sim = np.mean(sims)
        self.assertLess(mean_sim, 0.95, f"Random similarity too high ({mean_sim:.4f}), embedding space collapsed.")

    def test_9_degenerate_crops_handled_safely(self):
        """TEST 9: None and invalid crops return None without throwing exceptions."""
        if self.embedder is None:
            self.skipTest("Embedder not loaded.")
        self.assertIsNone(self.embedder.embed(None))
        self.assertIsNone(self.embedder.embed(np.zeros((0, 0, 3), dtype=np.uint8)))
        self.assertIsNone(self.embedder.embed(np.zeros((2, 2, 3), dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
