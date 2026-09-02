import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from src.core.types import FaceEmbedding
from src.face.alignment import align_face, ARCFACE_REF_LANDMARKS
from src.face.alignment_diagnostic import (
    inspect_onnx_model,
    preprocess_variant_a,
    preprocess_variant_b,
    preprocess_variant_c,
    preprocess_variant_d,
    evaluate_configuration,
)
from src.face.diagnostic import compute_pairwise_diagnostic
from src.face.embedder import FaceEmbedder
from src.face.preprocessing import l2_normalize


class TestArcFaceDiagnostic(unittest.TestCase):
    """Unit test suite for Slice 5.2: ArcFace Input Validation and Face Alignment."""

    @classmethod
    def setUpClass(cls):
        cls.embedder = FaceEmbedder()

    def test_1_model_input_inspection(self):
        """TEST 1: Model input inspection reports correct tensor shapes and dtypes."""
        info = inspect_onnx_model("models/arcface_resnet100.onnx")
        self.assertEqual(info.input_name, "data")
        self.assertEqual(info.input_shape, [1, 3, 112, 112])
        self.assertEqual(info.output_name, "fc1")
        self.assertEqual(info.output_shape, [1, 512])

    def test_2_preprocessing_shape(self):
        """TEST 2: Preprocessing variants produce NCHW (1, 3, 112, 112) float32 blobs."""
        crop = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)

        blob_a = preprocess_variant_a(crop)
        blob_b = preprocess_variant_b(crop)
        blob_c = preprocess_variant_c(crop)
        blob_d = preprocess_variant_d(crop)

        for b in [blob_a, blob_b, blob_c, blob_d]:
            self.assertEqual(b.shape, (1, 3, 112, 112))
            self.assertEqual(b.dtype, np.float32)

    def test_3_channel_conversion_and_normalization(self):
        """TEST 3: Verify preprocessing normalization ranges."""
        crop = np.full((112, 112, 3), 128, dtype=np.uint8)
        blob_c = preprocess_variant_c(crop)

        # Pixel value 128 maps to (128 - 127.5) / 127.5 = 0.5 / 127.5 = +0.00392
        self.assertAlmostEqual(float(np.mean(blob_c)), 0.00392, places=4)

    def test_4_alignment_output_shape(self):
        """TEST 4: 5-point facial landmark alignment produces 112x112 aligned face crops."""
        img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        landmarks = np.array([
            [60.0, 70.0],
            [130.0, 70.0],
            [95.0, 100.0],
            [70.0, 140.0],
            [120.0, 140.0],
        ], dtype=np.float32)

        aligned = align_face(img, landmarks, (112, 112))

        self.assertIsNotNone(aligned)
        self.assertEqual(aligned.shape, (112, 112, 3))
        self.assertEqual(aligned.dtype, np.uint8)

    def test_5_embeddings_are_finite(self):
        """TEST 5: Embeddings contain no NaN or Inf values."""
        dummy_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        emb = self.embedder.embed(dummy_crop)

        self.assertIsNotNone(emb)
        self.assertFalse(np.isnan(emb.vector).any())
        self.assertFalse(np.isinf(emb.vector).any())

    def test_6_embeddings_are_l2_normalized(self):
        """TEST 6: Embedding vector L2 norm is approximately 1.0."""
        dummy_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        emb = self.embedder.embed(dummy_crop)

        self.assertIsNotNone(emb)
        norm = np.linalg.norm(emb.vector)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_7_similarity_matrix_calculation(self):
        """TEST 7: Verify dot product cosine similarity logic on known normalized vectors."""
        v1 = l2_normalize(np.array([1.0, 0.0] + [0.0] * 510, dtype=np.float32))
        v2 = l2_normalize(np.array([1.0, 0.0] + [0.0] * 510, dtype=np.float32))
        v3 = l2_normalize(np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32))

        sim_identical = float(np.dot(v1, v2))
        sim_orthogonal = float(np.dot(v1, v3))

        self.assertAlmostEqual(sim_identical, 1.0, places=5)
        self.assertAlmostEqual(sim_orthogonal, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
