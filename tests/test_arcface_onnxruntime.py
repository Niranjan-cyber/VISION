import unittest
import numpy as np

from src.face.embedder import ONNXRuntimeArcFaceEmbedder, OpenCVArcFaceEmbedder


class TestONNXRuntimeArcFaceEmbedder(unittest.TestCase):
    """Unit test suite for Slice 5.3: ONNX Runtime ArcFace Embedder."""

    @classmethod
    def setUpClass(cls):
        cls.ort_embedder = ONNXRuntimeArcFaceEmbedder()
        cls.cv_embedder = OpenCVArcFaceEmbedder()

    def test_1_model_loads_successfully(self):
        """TEST 1: Model loads and session is initialized."""
        self.assertIsNotNone(self.ort_embedder.session)
        self.assertEqual(self.ort_embedder.input_name, "data")
        self.assertEqual(self.ort_embedder.output_name, "fc1")

    def test_2_input_shape_is_correct(self):
        """TEST 2: Input tensor shape matches [1, 3, 112, 112]."""
        self.assertEqual(self.ort_embedder.input_shape, [1, 3, 112, 112])

    def test_3_output_shape_is_512(self):
        """TEST 3: Output tensor shape matches [1, 512]."""
        self.assertEqual(self.ort_embedder.output_shape, [1, 512])

    def test_4_preprocessing_produces_correct_tensor(self):
        """TEST 4: Preprocessing produces [1, 3, 112, 112] float32 array."""
        dummy_crop = np.random.randint(0, 256, (90, 75, 3), dtype=np.uint8)
        tensor = self.ort_embedder.preprocess(dummy_crop)

        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (1, 3, 112, 112))
        self.assertEqual(tensor.dtype, np.float32)

    def test_5_output_dtype_is_float32(self):
        """TEST 5: Embedding output vector dtype is float32."""
        dummy_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        emb = self.ort_embedder.embed(dummy_crop)

        self.assertIsNotNone(emb)
        self.assertEqual(emb.vector.dtype, np.float32)
        self.assertEqual(emb.dimension, 512)
        self.assertEqual(len(emb.vector), 512)

    def test_6_output_is_finite(self):
        """TEST 6: Embedding contains only finite values (no NaNs or Infs)."""
        dummy_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        emb = self.ort_embedder.embed(dummy_crop)

        self.assertIsNotNone(emb)
        self.assertFalse(np.isnan(emb.vector).any())
        self.assertFalse(np.isinf(emb.vector).any())

    def test_7_output_is_l2_normalized(self):
        """TEST 7: Embedding output is strictly L2 normalized."""
        dummy_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        emb = self.ort_embedder.embed(dummy_crop)

        self.assertIsNotNone(emb)
        norm = np.linalg.norm(emb.vector)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_8_repeated_inference_is_deterministic(self):
        """TEST 8: Repeated inference on identical inputs produces identical vectors."""
        dummy_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)

        emb1 = self.ort_embedder.embed(dummy_crop)
        emb2 = self.ort_embedder.embed(dummy_crop)

        self.assertIsNotNone(emb1)
        self.assertIsNotNone(emb2)
        self.assertTrue(np.allclose(emb1.vector, emb2.vector, atol=1e-5))

    def test_9_different_inputs_produce_distinct_embeddings(self):
        """TEST 9: Different inputs do not produce identical memory references or exact duplicate arrays."""
        crop_a = np.zeros((112, 112, 3), dtype=np.uint8)
        crop_b = np.full((112, 112, 3), 255, dtype=np.uint8)

        emb_a = self.ort_embedder.embed(crop_a)
        emb_b = self.ort_embedder.embed(crop_b)

        self.assertIsNotNone(emb_a)
        self.assertIsNotNone(emb_b)
        self.assertIsNot(emb_a.vector, emb_b.vector)
        self.assertFalse(np.array_equal(emb_a.vector, emb_b.vector))

    def test_10_invalid_or_empty_crops_handled_safely(self):
        """TEST 10: Invalid, empty, or degenerate crops return None safely."""
        self.assertIsNone(self.ort_embedder.embed(None))
        self.assertIsNone(self.ort_embedder.embed(np.zeros((0, 0, 3), dtype=np.uint8)))
        self.assertIsNone(self.ort_embedder.embed(np.zeros((2, 2, 3), dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
