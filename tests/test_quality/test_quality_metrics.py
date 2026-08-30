"""
Unit tests for Quality Analyzer, Brightness, Blur, Contrast, and Resolution analyzers.
"""

import cv2
import numpy as np
import pytest

from src.quality.brightness import BrightnessAnalyzer
from src.quality.blur import BlurAnalyzer
from src.quality.contrast import ContrastAnalyzer
from src.quality.resolution import ResolutionAnalyzer
from src.quality.analyzer import QualityAnalyzer


@pytest.fixture
def bright_sharp_frame():
    # Synthetic clear, well-lit checkerboard with high contrast and sharpness
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    for y in range(0, 720, 40):
        for x in range(0, 1280, 40):
            if (x // 40 + y // 40) % 2 == 0:
                img[y:y+40, x:x+40] = [200, 200, 200]
            else:
                img[y:y+40, x:x+40] = [80, 80, 80]
    return img


@pytest.fixture
def dark_frame():
    # Very dark low-light frame
    img = np.full((720, 1280, 3), fill_value=25, dtype=np.uint8)
    return img


@pytest.fixture
def blurry_frame():
    # Heavily blurred frame
    img = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (51, 51), sigmaX=20)


@pytest.fixture
def low_res_frame():
    # Tiny low-res image
    return np.full((240, 320, 3), fill_value=128, dtype=np.uint8)


def test_brightness_analyzer(bright_sharp_frame, dark_frame):
    analyzer = BrightnessAnalyzer()
    
    res_bright = analyzer.analyze(bright_sharp_frame)
    assert not res_bright["is_low_light"]
    assert res_bright["mean_luminance"] > 70

    res_dark = analyzer.analyze(dark_frame)
    assert res_dark["is_low_light"]
    assert res_dark["low_light_score"] > 0.5


def test_blur_analyzer(bright_sharp_frame, blurry_frame):
    analyzer = BlurAnalyzer()

    res_sharp = analyzer.analyze(bright_sharp_frame)
    assert not res_sharp["is_blurry"]
    assert res_sharp["laplacian_var"] > 100

    res_blur = analyzer.analyze(blurry_frame)
    assert res_blur["is_blurry"]
    assert res_blur["blur_score"] > 0.6


def test_contrast_analyzer(bright_sharp_frame):
    analyzer = ContrastAnalyzer()
    res = analyzer.analyze(bright_sharp_frame)
    assert res["rms_contrast"] > 0.1
    assert res["dynamic_range"] > 50


def test_resolution_analyzer(bright_sharp_frame, low_res_frame):
    analyzer = ResolutionAnalyzer(min_target_width=1280, min_target_height=720)

    res_hd = analyzer.analyze(bright_sharp_frame)
    assert not res_hd["is_low_res"]
    assert res_hd["width"] == 1280

    res_low = analyzer.analyze(low_res_frame)
    assert res_low["is_low_res"]
    assert res_low["width"] == 320


def test_quality_analyzer_master(dark_frame, blurry_frame, bright_sharp_frame):
    qa = QualityAnalyzer()

    report_dark = qa.analyze_frame(dark_frame)
    assert not report_dark.is_good_quality
    assert "zero_dce" in report_dark.recommended_enhancements

    report_blur = qa.analyze_frame(blurry_frame)
    assert not report_blur.is_good_quality
    assert "rvrt" in report_blur.recommended_enhancements

    report_good = qa.analyze_frame(bright_sharp_frame)
    assert report_good.is_good_quality
    assert len(report_good.recommended_enhancements) == 0
