import unittest
import numpy as np

from src.core.types import BoundingBox, FaceEmbedding, IdentityMatch, Track
from src.face.gallery import FaceGallery
from src.face.matcher import FaceMatcher
from src.face.preprocessing import l2_normalize


class TestFaceMatchingModule(unittest.TestCase):
    """Unit test suite for Vertical Slice 5: Face Recognition & Identity Matching."""

    def test_1_identity_match_construction(self):
        """TEST 1: IdentityMatch construction."""
        match_known = IdentityMatch(
            identity="person_001",
            similarity=0.88,
            is_match=True,
            second_similarity=0.40,
            margin=0.48,
        )
        self.assertEqual(match_known.identity, "person_001")
        self.assertEqual(match_known.similarity, 0.88)
        self.assertTrue(match_known.is_match)
        self.assertEqual(match_known.second_similarity, 0.40)
        self.assertAlmostEqual(match_known.margin, 0.48)

        match_unknown = IdentityMatch(
            identity=None,
            similarity=0.45,
            is_match=False,
            second_similarity=0.40,
            margin=0.05,
        )
        self.assertIsNone(match_unknown.identity)
        self.assertEqual(match_unknown.similarity, 0.45)
        self.assertFalse(match_unknown.is_match)

    def test_2_gallery_accepts_valid_512d_embeddings(self):
        """TEST 2: Gallery accepts valid 512-D embeddings."""
        gallery = FaceGallery()
        vec = np.random.randn(512).astype(np.float32)
        norm_vec = l2_normalize(vec)
        embedding = FaceEmbedding(vector=norm_vec, dimension=512)

        gallery.add("person_001", embedding)
        self.assertEqual(len(gallery), 1)
        self.assertIn("person_001", gallery.identities())
        self.assertEqual(len(gallery.get("person_001")), 1)

    def test_3_gallery_rejects_invalid_dimensions(self):
        """TEST 3: Gallery rejects invalid dimensions."""
        gallery = FaceGallery()
        invalid_vec = np.random.randn(128).astype(np.float32)

        with self.assertRaises(ValueError):
            gallery.add("person_001", invalid_vec)

        with self.assertRaises(ValueError):
            gallery.add("person_001", FaceEmbedding(vector=invalid_vec, dimension=128))

    def test_4_gallery_supports_multiple_embeddings_per_identity(self):
        """TEST 4: Gallery supports multiple embeddings per identity."""
        gallery = FaceGallery()
        vec1 = l2_normalize(np.random.randn(512).astype(np.float32))
        vec2 = l2_normalize(np.random.randn(512).astype(np.float32))
        vec3 = l2_normalize(np.random.randn(512).astype(np.float32))

        gallery.add("person_001", vec1)
        gallery.add("person_001", vec2)
        gallery.add("person_002", vec3)

        self.assertEqual(len(gallery.get("person_001")), 2)
        self.assertEqual(len(gallery.get("person_002")), 1)
        self.assertEqual(len(gallery), 3)
        self.assertEqual(gallery.identities(), ["person_001", "person_002"])

    def test_5_cosine_similarity_identifies_highest_scoring_identity(self):
        """TEST 5: Cosine similarity correctly identifies the highest-scoring identity with margin."""
        gallery = FaceGallery()

        base_p1 = l2_normalize(np.array([1.0] + [0.0] * 511, dtype=np.float32))
        base_p2 = l2_normalize(np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32))

        gallery.add("person_001", base_p1)
        gallery.add("person_002", base_p2)

        matcher = FaceMatcher(gallery, threshold=0.70, margin=0.10)

        # Query vector identical to person_001
        query_vec = l2_normalize(np.array([1.0] + [0.0] * 511, dtype=np.float32))
        match = matcher.match(query_vec)

        self.assertEqual(match.identity, "person_001")
        self.assertTrue(match.is_match)
        self.assertAlmostEqual(match.similarity, 1.0, places=4)
        self.assertAlmostEqual(match.second_similarity, 0.0, places=4)
        self.assertAlmostEqual(match.margin, 1.0, places=4)

    def test_6_threshold_rejection(self):
        """TEST 6: Threshold rejection - score < threshold returns is_match=False and identity=None."""
        gallery = FaceGallery()
        vec = l2_normalize(np.array([1.0] + [0.0] * 511, dtype=np.float32))
        gallery.add("person_001", vec)

        matcher = FaceMatcher(gallery, threshold=0.70, margin=0.10)

        query_vec = l2_normalize(np.array([0.54, 0.841] + [0.0] * 510, dtype=np.float32))
        match = matcher.match(query_vec)

        self.assertIsNone(match.identity)
        self.assertFalse(match.is_match)
        self.assertAlmostEqual(match.similarity, 0.54, places=2)

    def test_7_exact_threshold_acceptance(self):
        """TEST 7: Similarity exactly equal to threshold -> is_match = True."""
        gallery = FaceGallery()

        ref_vec = l2_normalize(np.array([0.70, 0.7141428] + [0.0] * 510, dtype=np.float32))
        query_vec = l2_normalize(np.array([1.0] + [0.0] * 511, dtype=np.float32))

        gallery.add("person_001", ref_vec)

        matcher = FaceMatcher(gallery, threshold=0.70, margin=0.10)
        match = matcher.match(query_vec)

        self.assertEqual(match.identity, "person_001")
        self.assertTrue(match.is_match)

    def test_8_margin_rejection(self):
        """TEST 8: Margin rejection - best - second < margin returns is_match=False."""
        gallery = FaceGallery()

        ref_p1 = l2_normalize(np.array([0.80, 0.60] + [0.0] * 510, dtype=np.float32)) # sim 0.80
        ref_p2 = l2_normalize(np.array([0.75, 0.66] + [0.0] * 510, dtype=np.float32)) # sim 0.75

        gallery.add("person_001", ref_p1)
        gallery.add("person_002", ref_p2)

        matcher = FaceMatcher(gallery, threshold=0.70, margin=0.10)
        query = l2_normalize(np.array([1.0] + [0.0] * 511, dtype=np.float32))

        # Margin is 0.80 - 0.75 = 0.05 < 0.10 margin threshold -> Reject!
        match = matcher.match(query)
        self.assertIsNone(match.identity)
        self.assertFalse(match.is_match)
        self.assertAlmostEqual(match.margin, 0.05, places=2)

    def test_9_empty_gallery_returns_safe_unknown(self):
        """TEST 9: Empty gallery returns a safe Unknown result."""
        empty_gallery = FaceGallery()
        matcher = FaceMatcher(empty_gallery, threshold=0.60, margin=0.10)

        query_vec = l2_normalize(np.random.randn(512).astype(np.float32))
        match = matcher.match(query_vec)

        self.assertIsNone(match.identity)
        self.assertFalse(match.is_match)
        self.assertEqual(match.similarity, 0.0)

    def test_10_single_identity_gallery_handled_safely(self):
        """TEST 10: Single-identity gallery handled safely without margin errors."""
        gallery = FaceGallery()
        ref_vec = l2_normalize(np.array([0.80, 0.60] + [0.0] * 510, dtype=np.float32))
        gallery.add("person_solo", ref_vec)

        matcher = FaceMatcher(gallery, threshold=0.70, margin=0.10)
        query = l2_normalize(np.array([1.0] + [0.0] * 511, dtype=np.float32))

        match = matcher.match(query)
        self.assertEqual(match.identity, "person_solo")
        self.assertTrue(match.is_match)
        self.assertAlmostEqual(match.similarity, 0.80, places=2)
        self.assertAlmostEqual(match.second_similarity, 0.0, places=2)

    def test_11_nan_and_inf_embeddings_rejected_safely(self):
        """TEST 11: NaN/Inf embeddings rejected safely."""
        gallery = FaceGallery()
        ref_vec = l2_normalize(np.random.randn(512).astype(np.float32))
        gallery.add("person_001", ref_vec)

        matcher = FaceMatcher(gallery, threshold=0.60, margin=0.10)

        nan_vec = np.full(512, np.nan, dtype=np.float32)
        match_nan = matcher.match(nan_vec)
        self.assertIsNone(match_nan.identity)
        self.assertFalse(match_nan.is_match)

        inf_vec = np.full(512, np.inf, dtype=np.float32)
        match_inf = matcher.match(inf_vec)
        self.assertIsNone(match_inf.identity)
        self.assertFalse(match_inf.is_match)

    def test_12_query_embedding_not_mutated(self):
        """TEST 12: Incoming query embedding is not mutated."""
        gallery = FaceGallery()
        ref_vec = l2_normalize(np.random.randn(512).astype(np.float32))
        gallery.add("person_001", ref_vec)

        matcher = FaceMatcher(gallery, threshold=0.60, margin=0.10)

        raw_query = np.random.randn(512).astype(np.float32)
        raw_query_copy = np.copy(raw_query)

        _match = matcher.match(raw_query)
        self.assertTrue(np.array_equal(raw_query, raw_query_copy))


if __name__ == "__main__":
    unittest.main()
