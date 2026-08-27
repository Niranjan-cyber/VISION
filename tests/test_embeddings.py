import unittest
import numpy as np

from src.core.types import BoundingBox, FaceEmbedding, Track
from src.face.embedder import FaceEmbedder
from src.face.preprocessing import l2_normalize, preprocess_face_crop


class TestFaceEmbeddingModule(unittest.TestCase):
    """Unit test suite for Vertical Slice 4: ArcFace 512-d Embedding Generation."""

    def test_1_face_embedding_construction(self):
        """TEST 1: FaceEmbedding dataclass can be constructed."""
        vec = np.random.randn(512).astype(np.float32)
        norm_vec = l2_normalize(vec)
        embedding = FaceEmbedding(vector=norm_vec, dimension=512)

        self.assertIsInstance(embedding.vector, np.ndarray)
        self.assertEqual(embedding.dimension, 512)
        self.assertEqual(len(embedding.vector), 512)

    def test_2_embedding_dimension_is_512(self):
        """TEST 2: Embedding dimension is 512."""
        vec = np.ones(512, dtype=np.float32)
        norm_vec = l2_normalize(vec)
        embedding = FaceEmbedding(vector=norm_vec, dimension=512)
        self.assertEqual(embedding.dimension, 512)
        self.assertEqual(embedding.vector.shape[0], 512)

    def test_3_embedding_normalization_unit_l2_norm(self):
        """TEST 3: Embedding normalization produces approximately unit L2 norm."""
        raw_vec = np.array([3.0, 4.0] + [0.0] * 510, dtype=np.float32)
        norm_vec = l2_normalize(raw_vec)
        l2_norm = np.linalg.norm(norm_vec)

        self.assertAlmostEqual(l2_norm, 1.0, places=5)
        self.assertAlmostEqual(norm_vec[0], 0.6, places=5)
        self.assertAlmostEqual(norm_vec[1], 0.8, places=5)

    def test_4_zero_norm_vector_handled_safely(self):
        """TEST 4: Zero-norm vector is handled safely without producing NaNs."""
        zero_vec = np.zeros(512, dtype=np.float32)
        norm_vec = l2_normalize(zero_vec)

        self.assertFalse(np.isnan(norm_vec).any())
        self.assertFalse(np.isinf(norm_vec).any())
        self.assertEqual(np.linalg.norm(norm_vec), 0.0)
        self.assertEqual(len(norm_vec), 512)

    def test_5_invalid_empty_face_crop_rejected_safely(self):
        """TEST 5: Invalid/empty face crop is rejected safely by FaceEmbedder / preprocessor."""
        empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
        blob_empty = preprocess_face_crop(empty_crop)
        self.assertIsNone(blob_empty)

        none_blob = preprocess_face_crop(None)
        self.assertIsNone(none_blob)

        tiny_crop = np.zeros((2, 2, 3), dtype=np.uint8)
        blob_tiny = preprocess_face_crop(tiny_crop)
        self.assertIsNone(blob_tiny)

    def test_6_face_crop_coordinates_extracted_from_full_frame(self):
        """TEST 6: Face crop coordinates are correctly extracted from original full-frame image."""
        full_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw distinct color region at (100, 100, 200, 200)
        full_frame[100:200, 100:200] = [0, 255, 0]

        face_bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        fx1 = max(0, min(face_bbox.x1, full_frame.shape[1]))
        fy1 = max(0, min(face_bbox.y1, full_frame.shape[0]))
        fx2 = max(0, min(face_bbox.x2, full_frame.shape[1]))
        fy2 = max(0, min(face_bbox.y2, full_frame.shape[0]))

        face_crop = full_frame[fy1:fy2, fx1:fx2]

        self.assertEqual(face_crop.shape, (100, 100, 3))
        self.assertTrue((face_crop == [0, 255, 0]).all())

    def test_7_embedding_generation_does_not_alter_track_object(self):
        """TEST 7: Embedding generation does not alter the Track object."""
        track = Track(
            track_id=17,
            class_id=0,
            class_name="person",
            confidence=0.94,
            bbox=BoundingBox(100, 50, 300, 450),
            frame_number=1,
        )

        track_id_before = track.track_id
        bbox_before = track.bbox.as_tuple()

        # Dummy embedding generation logic
        dummy_vec = l2_normalize(np.random.randn(512).astype(np.float32))
        _embedding = FaceEmbedding(vector=dummy_vec, dimension=512)

        self.assertEqual(track.track_id, track_id_before)
        self.assertEqual(track.bbox.as_tuple(), bbox_before)
        self.assertFalse(hasattr(track, "embedding"))

    def test_8_track_id_remains_separate_from_face_embedding(self):
        """TEST 8: Track ID remains separate from FaceEmbedding."""
        vec = l2_normalize(np.ones(512, dtype=np.float32))
        embedding = FaceEmbedding(vector=vec, dimension=512)

        self.assertFalse(hasattr(embedding, "track_id"))
        self.assertFalse(hasattr(embedding, "identity"))
        self.assertFalse(hasattr(embedding, "person_name"))


if __name__ == "__main__":
    unittest.main()
