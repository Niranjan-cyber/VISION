import os
import unittest
import numpy as np

from src.core.types import BoundingBox, FaceDetection
from src.face.alignment import align_face
from src.face.detector import FaceDetector
from src.face.gallery import FaceGallery, load_gallery_from_dir
from src.face.matcher import FaceMatcher
from src.face.modern_embedder import W600KR50Embedder
from src.face.w600k_preprocessing import preprocess_w600k_crop


class TestSlice56Integration(unittest.TestCase):
    """Unit test suite for Slice 5.6: W600K-R50 and 5-Point Landmark Alignment Integration."""

    @classmethod
    def setUpClass(cls):
        cls.model_path = "models/w600k_r50.onnx"
        cls.embedder = W600KR50Embedder(cls.model_path)
        cls.detector = FaceDetector(score_threshold=0.50)

    def test_1_w600k_model_loading_and_contract(self):
        """TEST 1: W600K-R50 session loads with expected input/output contract."""
        self.assertIsNotNone(self.embedder.session)
        self.assertEqual(self.embedder.embedding_dim, 512)
        self.assertEqual(self.embedder.input_name, "input.1")
        self.assertEqual(self.embedder.output_name, "683")

    def test_2_embedding_dimension_and_normalization(self):
        """TEST 2: Generated embedding is 512-D and strictly L2-normalized."""
        dummy_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        emb = self.embedder.embed(dummy_crop)

        self.assertIsNotNone(emb)
        self.assertEqual(emb.dimension, 512)
        self.assertEqual(len(emb.vector), 512)
        norm = np.linalg.norm(emb.vector)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_3_landmark_alignment_output_shape(self):
        """TEST 3: Landmark alignment produces exact (112, 112, 3) aligned crop."""
        dummy_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        dummy_landmarks = np.array([
            [100, 100],  # right eye
            [150, 100],  # left eye
            [125, 130],  # nose
            [110, 160],  # right mouth
            [140, 160],  # left mouth
        ], dtype=np.float32)

        aligned = align_face(dummy_img, dummy_landmarks, target_size=(112, 112))
        self.assertIsNotNone(aligned)
        self.assertEqual(aligned.shape, (112, 112, 3))
        self.assertEqual(aligned.dtype, np.uint8)

    def test_4_gallery_uses_alignment(self):
        """TEST 4: load_gallery_from_dir successfully enrolls aligned gallery embeddings."""
        gallery = load_gallery_from_dir("data/face_gallery", self.detector, self.embedder)
        self.assertFalse(gallery.is_empty())
        self.assertIn("Atharva_Jaysingpure", gallery.identities())
        self.assertIn("Shreyas_Chavan", gallery.identities())

        # Each enrolled embedding must be 512-D and L2-normalized
        for id_name in gallery.identities():
            for emb in gallery.get(id_name):
                self.assertEqual(len(emb), 512)
                self.assertAlmostEqual(np.linalg.norm(emb), 1.0, places=5)

    def test_5_aligned_recognition_discrimination(self):
        """TEST 5: Aligned embeddings demonstrate clear separation between genuine and impostor matches."""
        gallery = load_gallery_from_dir("data/face_gallery", self.detector, self.embedder)
        matcher = FaceMatcher(gallery, threshold=0.60, margin=0.10)

        # Test with Shreyas front aligned crop
        shr_path = "data/face_gallery/Shreyas_Chavan/front.jpeg"
        if os.path.exists(shr_path):
            import cv2
            img = cv2.imread(shr_path)
            faces = self.detector.detect(img)
            self.assertTrue(len(faces) > 0)
            best_face = max(faces, key=lambda f: f.confidence)
            aligned = align_face(img, best_face.landmarks)
            self.assertIsNotNone(aligned)

            emb = self.embedder.embed(aligned)
            match = matcher.match(emb)
            self.assertTrue(match.is_match)
            self.assertEqual(match.identity, "Shreyas_Chavan")
            self.assertGreater(match.similarity, 0.60)
            self.assertGreater(match.margin, 0.10)


if __name__ == "__main__":
    unittest.main()
