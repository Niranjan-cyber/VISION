"""
Tests for src/core/device.py's device/provider resolution logic. Uses
mocks so both the "CUDA available" and "CUDA unavailable" branches are
covered deterministically regardless of what hardware the test runner has
— a real end-to-end confirmation that GPU is actually used when present is
covered separately in test_pipeline_session.py (skipped when no GPU).
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.core.device as device_mod


def _reset_cache():
    device_mod._torch_cuda_available = None
    device_mod._gpu_name = None
    device_mod._ort_cuda_available = None
    device_mod._banner_printed = False


class TestResolveYoloDevice(unittest.TestCase):
    def setUp(self):
        _reset_cache()

    def test_cpu_forced_regardless_of_hardware(self):
        with patch.object(device_mod, "torch_cuda_available", return_value=True):
            self.assertEqual(device_mod.resolve_yolo_device("cpu"), "cpu")

    def test_auto_uses_cuda_when_available(self):
        with patch.object(device_mod, "torch_cuda_available", return_value=True):
            self.assertEqual(device_mod.resolve_yolo_device("auto"), "cuda:0")

    def test_auto_falls_back_to_cpu_when_unavailable(self):
        with patch.object(device_mod, "torch_cuda_available", return_value=False):
            self.assertEqual(device_mod.resolve_yolo_device("auto"), "cpu")

    def test_cuda_requested_and_available(self):
        with patch.object(device_mod, "torch_cuda_available", return_value=True):
            self.assertEqual(device_mod.resolve_yolo_device("cuda"), "cuda:0")

    def test_cuda_requested_but_unavailable_falls_back_with_warning(self):
        with patch.object(device_mod, "torch_cuda_available", return_value=False):
            self.assertEqual(device_mod.resolve_yolo_device("cuda"), "cpu")

    def test_invalid_preference_treated_as_auto(self):
        with patch.object(device_mod, "torch_cuda_available", return_value=False):
            self.assertEqual(device_mod.resolve_yolo_device("not-a-real-device"), "cpu")


class TestResolveOrtProviders(unittest.TestCase):
    def setUp(self):
        _reset_cache()

    def test_cpu_forced_regardless_of_hardware(self):
        with patch.object(device_mod, "ort_cuda_available", return_value=True):
            self.assertEqual(device_mod.resolve_ort_providers("cpu"), ["CPUExecutionProvider"])

    def test_auto_uses_cuda_when_available(self):
        with patch.object(device_mod, "ort_cuda_available", return_value=True):
            providers = device_mod.resolve_ort_providers("auto")
            self.assertEqual(providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def test_auto_falls_back_to_cpu_when_unavailable(self):
        with patch.object(device_mod, "ort_cuda_available", return_value=False):
            self.assertEqual(device_mod.resolve_ort_providers("auto"), ["CPUExecutionProvider"])

    def test_cuda_requested_but_unavailable_falls_back_with_warning(self):
        with patch.object(device_mod, "ort_cuda_available", return_value=False):
            self.assertEqual(device_mod.resolve_ort_providers("cuda"), ["CPUExecutionProvider"])

    def test_cuda_requested_and_available(self):
        with patch.object(device_mod, "ort_cuda_available", return_value=True):
            providers = device_mod.resolve_ort_providers("cuda")
            self.assertEqual(providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])


class TestHardwareDetectionReflectsRealEnvironment(unittest.TestCase):
    """These call the real (uncached) detection functions — they must never
    raise, and must return types consistent with whatever this machine
    actually has, whether or not a GPU/CUDA build is present."""

    def setUp(self):
        _reset_cache()

    def test_torch_cuda_available_returns_bool(self):
        self.assertIsInstance(device_mod.torch_cuda_available(), bool)

    def test_ort_cuda_available_returns_bool(self):
        self.assertIsInstance(device_mod.ort_cuda_available(), bool)

    def test_gpu_name_is_none_or_string(self):
        name = device_mod.gpu_name()
        self.assertTrue(name is None or isinstance(name, str))


class TestPrintHardwareBanner(unittest.TestCase):
    def setUp(self):
        _reset_cache()

    def test_prints_once_per_process_by_default(self):
        with patch("builtins.print") as mock_print:
            device_mod.print_hardware_banner("auto", "cpu", "CPUExecutionProvider")
            device_mod.print_hardware_banner("auto", "cpu", "CPUExecutionProvider")
            self.assertEqual(mock_print.call_count, 1)

    def test_force_reprints(self):
        with patch("builtins.print") as mock_print:
            device_mod.print_hardware_banner("auto", "cpu", "CPUExecutionProvider")
            device_mod.print_hardware_banner("auto", "cpu", "CPUExecutionProvider", force=True)
            self.assertEqual(mock_print.call_count, 2)

    def test_never_claims_cuda_for_a_cpu_device_string(self):
        with patch("builtins.print") as mock_print:
            device_mod.print_hardware_banner("auto", "cpu", "CPUExecutionProvider")
            printed = mock_print.call_args[0][0]
            self.assertIn("YOLO (detection)", printed)
            yolo_line = [l for l in printed.splitlines() if "YOLO (detection)" in l][0]
            self.assertIn("CPU", yolo_line)
            self.assertNotIn("CUDA", yolo_line)


if __name__ == "__main__":
    unittest.main()
