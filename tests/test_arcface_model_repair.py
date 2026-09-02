import os
import unittest
import numpy as np
import onnx
import onnxruntime as ort

from src.face.model_integrity_diagnostic import (
    OFFICIAL_SHA256,
    OFFICIAL_FILE_SIZE,
    compute_file_sha256,
    inspect_model_integrity,
    create_patched_model,
)


class TestArcFaceModelRepair(unittest.TestCase):
    """Unit test suite for Slice 5.4: ArcFace ONNX Model Integrity & Repair."""

    @classmethod
    def setUpClass(cls):
        cls.orig_model_path = "models/arcface_resnet100.onnx"
        cls.patched_model_path = "models/arcface_resnet100_patched.onnx"

    def test_1_model_sha256_hash_verification(self):
        """TEST 1: Verify model SHA256 hash matches official ONNX Model Zoo hash."""
        self.assertTrue(os.path.exists(self.orig_model_path))
        computed_hash = compute_file_sha256(self.orig_model_path)
        self.assertEqual(computed_hash.lower(), OFFICIAL_SHA256.lower())

    def test_2_model_file_size_verification(self):
        """TEST 2: Verify model file size matches official expected bytes."""
        file_size = os.path.getsize(self.orig_model_path)
        self.assertEqual(file_size, OFFICIAL_FILE_SIZE)

    def test_3_batchnorm_spatial_detection(self):
        """TEST 3: Inspect BatchNorm nodes and verify spatial attributes."""
        info = inspect_model_integrity(self.orig_model_path)
        self.assertEqual(info.bn_nodes_count, 154)
        self.assertEqual(info.bn_spatial_0_count, 0)
        self.assertEqual(info.bn_spatial_1_count, 154)

    def test_4_patched_model_creation_and_validity(self):
        """TEST 4: Create patched model copy and verify with onnx.checker."""
        changed, total = create_patched_model(
            self.orig_model_path, self.patched_model_path
        )
        self.assertTrue(os.path.exists(self.patched_model_path))
        patched_model = onnx.load(self.patched_model_path)
        # Should not raise exception
        onnx.checker.check_model(patched_model)

    def test_5_original_vs_patched_inference_execution(self):
        """TEST 5: Verify both models execute successfully and output 512-D float32 tensors."""
        sess_orig = ort.InferenceSession(self.orig_model_path, providers=["CPUExecutionProvider"])
        sess_patch = ort.InferenceSession(self.patched_model_path, providers=["CPUExecutionProvider"])

        dummy_tensor = np.random.randn(1, 3, 112, 112).astype(np.float32)
        out_o = sess_orig.run(["fc1"], {"data": dummy_tensor})[0].flatten()
        out_p = sess_patch.run(["fc1"], {"data": dummy_tensor})[0].flatten()

        self.assertEqual(len(out_o), 512)
        self.assertEqual(len(out_p), 512)
        self.assertFalse(np.isnan(out_o).any())
        self.assertFalse(np.isnan(out_p).any())

    def test_6_random_input_embedding_properties(self):
        """TEST 6: Verify embeddings on random inputs are finite and non-trivial."""
        sess_orig = ort.InferenceSession(self.orig_model_path, providers=["CPUExecutionProvider"])
        dummy_tensor = np.random.randn(1, 3, 112, 112).astype(np.float32)
        out = sess_orig.run(["fc1"], {"data": dummy_tensor})[0].flatten()
        norm = np.linalg.norm(out)

        self.assertGreater(norm, 0.0)
        self.assertFalse(np.isinf(norm))


if __name__ == "__main__":
    unittest.main()
