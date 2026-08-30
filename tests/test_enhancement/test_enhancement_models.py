"""
Unit tests for Zero-DCE, Real-ESRGAN, BasicVSR, RVRT, and EnhancementManager.
"""

import numpy as np
import pytest
import torch

from src.enhancement.low_light.zero_dce import ZeroDCEEnhancer
from src.enhancement.low_resolution.realesrgan import RealESRGANEnhancer
from src.enhancement.low_resolution.basicvsr import BasicVSREnhancer
from src.enhancement.blur.rvrt import RVRTEnhancer
from src.enhancement.manager import EnhancementManager


@pytest.fixture
def dummy_dark_frame():
    return np.full((128, 128, 3), fill_value=20, dtype=np.uint8)


@pytest.fixture
def dummy_small_frame():
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


def test_zero_dce_enhancement(dummy_dark_frame):
    enhancer = ZeroDCEEnhancer(auto_download=False)
    enhanced = enhancer.enhance(dummy_dark_frame)

    assert enhanced.shape == dummy_dark_frame.shape
    assert enhanced.dtype == np.uint8
    # Enhanced frame should have significantly higher mean brightness
    assert np.mean(enhanced) > np.mean(dummy_dark_frame)


def test_realesrgan_enhancement(dummy_small_frame):
    enhancer = RealESRGANEnhancer(scale=2, auto_download=False)
    enhanced = enhancer.enhance(dummy_small_frame)

    h, w = dummy_small_frame.shape[:2]
    assert enhanced.shape == (h * 2, w * 2, 3)
    assert enhanced.dtype == np.uint8


def test_basicvsr_enhancement(dummy_small_frame):
    enhancer = BasicVSREnhancer(scale=2)
    seq = [dummy_small_frame, dummy_small_frame, dummy_small_frame]
    enhanced_seq = enhancer.enhance_sequence(seq)

    assert len(enhanced_seq) == 3
    h, w = dummy_small_frame.shape[:2]
    for ef in enhanced_seq:
        assert ef.shape == (h * 2, w * 2, 3)


def test_rvrt_enhancement(dummy_dark_frame):
    enhancer = RVRTEnhancer()
    enhanced = enhancer.enhance(dummy_dark_frame)

    assert enhanced.shape == dummy_dark_frame.shape
    assert enhanced.dtype == np.uint8


def test_enhancement_manager(dummy_dark_frame, tmp_path):
    manager = EnhancementManager(auto_download_weights=False)
    enhanced, report = manager.process_frame(dummy_dark_frame, mode="auto")

    assert enhanced is not None
    assert report.is_low_light
    assert np.mean(enhanced) > np.mean(dummy_dark_frame)
