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
        match_known = IdentityMatch(identity="person_001", similarity=0.88, is_match=True)
        self.assertEqual(match_known.identity, "person_001")
        self.assertEqual(match_known.similarity, 0.88)
        self.assertTrue(match_known.is_match)

        match_unknown = IdentityMatch(identity=None, similarity=0.45, is_match=False)
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
        """TEST 5: Cosine similarity correctly identifies the highest-scoring identity."""
        gallery = FaceGallery()

        # Create known base vector for person_001
        base_p1 = l2_normalize(np.ones(512, dtype=np.float32))
        base_p2 = l2_normalize(-np.ones(512, dtype=np.float32))

        gallery.add("person_001", base_p1)
        gallery.add("person_002", base_p2)

        matcher = FaceMatcher(gallery, threshold=0.70)

        # Query vector almost identical to person_001
        query_vec = l2_normalize(base_p1 + np.random.randn(512).astype(np.float32) * 0.01)
        match = matcher.match(query_vec)

        self.assertEqual(match.identity, "person_001")
        self.assertTrue(match.is_match)
        self.assertGreaterEqual(match.similarity, 0.70)

    def test_6_threshold_behavior_unknown_below_threshold(self):
        """TEST 6: Threshold behavior - below threshold returns is_match=False and identity=None."""
        gallery = FaceGallery()
        vec = l2_normalize(np.array([1.0] + [0.0] * 511, dtype=np.float32))
        gallery.add("person_001", vec)

        matcher = FaceMatcher(gallery, threshold=0.70)

        # Query vector with low similarity (~0.54)
        query_vec = l2_normalize(np.array([0.54] + [0.841] + [0.0] * 510, dtype=np.float32))
        match = matcher.match(query_vec)

        self.assertIsNone(match.identity)
        self.assertFalse(match.is_match)
        self.assertAlmostEqual(match.similarity, 0.54, places=2)

    def test_7_similarity_equal_to_threshold_is_match(self):
        """TEST 7: Similarity exactly equal to threshold -> is_match = True."""
        gallery = FaceGallery()

        ref_vec = l2_normalize(np.array([0.70, 0.7141428] + [0.0] * 510, dtype=np.float32))
        query_vec = l2_normalize(np.array([1.0] + [0.0] * 511, dtype=np.float32))
        # Dot product = 0.70

        gallery.add("person_001", ref_vec)

        matcher = FaceMatcher(gallery, threshold=0.70)
        match = matcher.match(query_vec)

        self.assertEqual(match.identity, "person_001")
        self.assertTrue(match.is_match)

    def test_8_empty_gallery_returns_safe_unknown(self):
        """TEST 8: Empty gallery returns a safe Unknown result."""
        empty_gallery = FaceGallery()
        matcher = FaceMatcher(empty_gallery, threshold=0.60)

        query_vec = l2_normalize(np.random.randn(512).astype(np.float32))
        match = matcher.match(query_vec)

        self.assertIsNone(match.identity)
        self.assertFalse(match.is_match)
        self.assertEqual(match.similarity, 0.0)

    def test_9_multiple_reference_embeddings_uses_strongest_similarity(self):
        """TEST 9: Identity with multiple reference embeddings uses strongest valid similarity."""
        gallery = FaceGallery()

        ref_p1_a = l2_normalize(np.array([0.5, 0.5] + [0.0] * 510, dtype=np.float32))
        ref_p1_b = l2_normalize(np.array([1.0, 0.0] + [0.0] * 510, dtype=np.float32)) # Strong match for query [1, 0, ...]

        gallery.add("person_001", ref_p1_a)
        gallery.add("person_001", ref_p1_b)

        matcher = FaceMatcher(gallery, threshold=0.60)
        query = l2_normalize(np.array([1.0, 0.0] + [0.0] * 510, dtype=np.float32))

        match = matcher.match(query)
        self.assertEqual(match.identity, "person_001")
        self.assertAlmostEqual(match.similarity, 1.0, places=4)
        self.assertTrue(match.is_match)

    def test_10_incoming_query_embedding_not_mutated(self):
        """TEST 10: Incoming query embedding is not mutated."""
        gallery = FaceGallery()
        ref_vec = l2_normalize(np.random.randn(512).astype(np.float32))
        gallery.add("person_001", ref_vec)

        matcher = FaceMatcher(gallery, threshold=0.60)

        raw_query = np.random.randn(512).astype(np.float32)
        raw_query_copy = np.copy(raw_query)

        _match = matcher.match(raw_query)
        self.assertTrue(np.array_equal(raw_query, raw_query_copy))

    def test_11_gallery_embeddings_remain_normalized(self):
        """TEST 11: Gallery embeddings remain normalized."""
        gallery = FaceGallery()
        unnormalized_vec = np.array([3.0, 4.0] + [0.0] * 510, dtype=np.float32)

        gallery.add("person_001", unnormalized_vec)
        stored_embeds = gallery.get("person_001")

        self.assertEqual(len(stored_embeds), 1)
        self.assertAlmostEqual(np.linalg.norm(stored_embeds[0]), 1.0, places=5)

    def test_12_track_id_remains_separate_from_identity_matching(self):
        """TEST 12: Track.track_id remains completely separate from identity matching."""
        track = Track(
            track_id=42,
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=BoundingBox(10, 10, 100, 200),
            frame_number=1,
        )

        match = IdentityMatch(identity="person_001", similarity=0.91, is_match=True)

        self.assertFalse(hasattr(track, "identity"))
        self.assertFalse(hasattr(track, "similarity"))
        self.assertFalse(hasattr(match, "track_id"))
        self.assertEqual(track.track_id, 42)


if __name__ == "__main__":
    unittest.main()
