import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock
import cv2
import numpy as np

from src.core.types import BoundingBox, FaceDetection, FaceEmbedding
from src.face.gallery import FaceGallery, load_gallery_from_dir


class TestGalleryLoadingModule(unittest.TestCase):
    """Unit test suite for FaceGallery image discovery and loading architecture."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_detector = MagicMock()
        self.mock_embedder = MagicMock()

        # Dummy face detection result
        self.dummy_face = FaceDetection(
            bbox=BoundingBox(10, 10, 50, 50),
            confidence=0.95,
        )
        self.mock_detector.detect.return_value = [self.dummy_face]

        # Dummy 512-d embedding result
        vec = np.ones(512, dtype=np.float32) / np.sqrt(512)
        self.dummy_embedding = FaceEmbedding(vector=vec, dimension=512)
        self.mock_embedder.embed.return_value = self.dummy_embedding

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_dummy_image(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        success, buf = cv2.imencode(".jpg", img)
        if success:
            with open(path, "wb") as f:
                f.write(buf.tobytes())
        else:
            with open(path, "wb") as f:
                f.write(b"dummy_image_data")

    def test_1_jpg_files_discovered(self):
        """TEST 1: .jpg files are discovered."""
        img_path = os.path.join(self.temp_dir, "person_a", "image1.jpg")
        self._create_dummy_image(img_path)

        gallery = load_gallery_from_dir(self.temp_dir, self.mock_detector, self.mock_embedder)
        self.assertEqual(len(gallery.identities()), 1)
        self.assertIn("person_a", gallery.identities())
        self.assertEqual(len(gallery.get("person_a")), 1)

    def test_2_uppercase_JPG_files_discovered(self):
        """TEST 2: .JPG files are discovered."""
        img_path = os.path.join(self.temp_dir, "person_a", "IMAGE1.JPG")
        self._create_dummy_image(img_path)

        gallery = load_gallery_from_dir(self.temp_dir, self.mock_detector, self.mock_embedder)
        self.assertEqual(len(gallery.identities()), 1)
        self.assertEqual(len(gallery.get("person_a")), 1)

    def test_3_jpeg_and_JPEG_files_discovered(self):
        """TEST 3: .jpeg / .JPEG files are discovered."""
        img1 = os.path.join(self.temp_dir, "person_b", "img1.jpeg")
        img2 = os.path.join(self.temp_dir, "person_b", "img2.JPEG")
        self._create_dummy_image(img1)
        self._create_dummy_image(img2)

        gallery = load_gallery_from_dir(self.temp_dir, self.mock_detector, self.mock_embedder)
        self.assertEqual(len(gallery.get("person_b")), 2)

    def test_4_png_and_PNG_files_discovered(self):
        """TEST 4: .png / .PNG files are discovered."""
        img1 = os.path.join(self.temp_dir, "person_c", "photo1.png")
        img2 = os.path.join(self.temp_dir, "person_c", "PHOTO2.PNG")
        self._create_dummy_image(img1)
        self._create_dummy_image(img2)

        gallery = load_gallery_from_dir(self.temp_dir, self.mock_detector, self.mock_embedder)
        self.assertEqual(len(gallery.get("person_c")), 2)

    def test_5_webp_and_WEBP_files_discovered(self):
        """TEST 5: .webp / .WEBP files are discovered."""
        img1 = os.path.join(self.temp_dir, "person_d", "face1.webp")
        img2 = os.path.join(self.temp_dir, "person_d", "FACE2.WEBP")
        self._create_dummy_image(img1)
        self._create_dummy_image(img2)

        gallery = load_gallery_from_dir(self.temp_dir, self.mock_detector, self.mock_embedder)
        self.assertEqual(len(gallery.get("person_d")), 2)

    def test_6_jfif_and_JFIF_files_discovered(self):
        """TEST 6: .jfif / .JFIF files are discovered."""
        img1 = os.path.join(self.temp_dir, "person_e", "shot1.jfif")
        img2 = os.path.join(self.temp_dir, "person_e", "SHOT2.JFIF")
        self._create_dummy_image(img1)
        self._create_dummy_image(img2)

        gallery = load_gallery_from_dir(self.temp_dir, self.mock_detector, self.mock_embedder)
        self.assertEqual(len(gallery.get("person_e")), 2)

    def test_7_unsupported_extensions_ignored(self):
        """TEST 7: Unsupported extensions such as .txt or .csv are ignored."""
        txt_file = os.path.join(self.temp_dir, "person_f", "notes.txt")
        os.makedirs(os.path.dirname(txt_file), exist_ok=True)
        with open(txt_file, "w") as f:
            f.write("text content")

        gallery = load_gallery_from_dir(self.temp_dir, self.mock_detector, self.mock_embedder)
        self.assertTrue(gallery.is_empty())

    def test_8_directories_inside_identity_folder_ignored(self):
        """TEST 8: Directories inside an identity folder are ignored."""
        nested_dir = os.path.join(self.temp_dir, "person_g", "subfolder")
        os.makedirs(nested_dir, exist_ok=True)

        gallery = load_gallery_from_dir(self.temp_dir, self.mock_detector, self.mock_embedder)
        self.assertTrue(gallery.is_empty())

    def test_9_valid_image_passes_detection_and_embedding(self):
        """TEST 9: A valid gallery image is passed through face detection and then ArcFace embedding generation."""
        img_path = os.path.join(self.temp_dir, "person_h", "ref.jpg")
        self._create_dummy_image(img_path)

        gallery = load_gallery_from_dir(self.temp_dir, self.mock_detector, self.mock_embedder)

        self.mock_detector.detect.assert_called_once()
        self.mock_embedder.embed.assert_called_once()
        self.assertEqual(len(gallery.get("person_h")), 1)

    def test_10_multiple_images_produce_multiple_reference_embeddings(self):
        """TEST 10: An identity with multiple images produces multiple reference embeddings."""
        img1 = os.path.join(self.temp_dir, "person_i", "img1.jpg")
        img2 = os.path.join(self.temp_dir, "person_i", "img2.png")
        img3 = os.path.join(self.temp_dir, "person_i", "img3.webp")
        self._create_dummy_image(img1)
        self._create_dummy_image(img2)
        self._create_dummy_image(img3)

        gallery = load_gallery_from_dir(self.temp_dir, self.mock_detector, self.mock_embedder)
        self.assertEqual(len(gallery.get("person_i")), 3)

    def test_11_unreadable_image_skipped_safely(self):
        """TEST 11: An unreadable image file is skipped safely."""
        corrupt_file = os.path.join(self.temp_dir, "person_j", "corrupt.jpg")
        os.makedirs(os.path.dirname(corrupt_file), exist_ok=True)
        with open(corrupt_file, "w") as f:
            f.write("not a real image")

        gallery = load_gallery_from_dir(self.temp_dir, self.mock_detector, self.mock_embedder)
        self.assertTrue(gallery.is_empty())

    def test_12_image_with_no_detected_face_skipped_safely(self):
        """TEST 12: An image with no detected face is skipped safely."""
        self.mock_detector.detect.return_value = []
        img_path = os.path.join(self.temp_dir, "person_k", "noface.jpg")
        self._create_dummy_image(img_path)

        gallery = load_gallery_from_dir(self.temp_dir, self.mock_detector, self.mock_embedder)
        self.assertTrue(gallery.is_empty())

    def test_13_missing_gallery_directory_returns_empty_gallery(self):
        """TEST 13: Missing gallery directory returns an empty FaceGallery."""
        missing_dir = os.path.join(self.temp_dir, "non_existent_folder")
        gallery = load_gallery_from_dir(missing_dir, self.mock_detector, self.mock_embedder)
        self.assertTrue(gallery.is_empty())

    def test_14_empty_gallery_directory_returns_empty_gallery(self):
        """TEST 14: Empty gallery directory returns an empty FaceGallery."""
        empty_dir = os.path.join(self.temp_dir, "empty_folder")
        os.makedirs(empty_dir, exist_ok=True)
        gallery = load_gallery_from_dir(empty_dir, self.mock_detector, self.mock_embedder)
        self.assertTrue(gallery.is_empty())


if __name__ == "__main__":
    unittest.main()
