"""
Brightness and low-light quality analysis module for CCTV surveillance frames.
"""

import cv2
import numpy as np
from typing import Dict, Any


class BrightnessAnalyzer:
    """
    Analyzes frame luminance and exposure to determine low-light degradation.
    """

    def __init__(
        self,
        dark_luminance_threshold: float = 60.0,
        dark_pixel_ratio_threshold: float = 0.50,
        extreme_dark_threshold: float = 25.0,
    ):
        """
        Args:
            dark_luminance_threshold: Mean luminance below which frame is flagged as low-light (0-255).
            dark_pixel_ratio_threshold: Fraction of pixels below extreme_dark_threshold.
            extreme_dark_threshold: Pixel luminance considered pitch dark.
        """
        self.dark_luminance_threshold = dark_luminance_threshold
        self.dark_pixel_ratio_threshold = dark_pixel_ratio_threshold
        self.extreme_dark_threshold = extreme_dark_threshold

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze brightness and illumination properties of a BGR frame.

        Returns:
            Dict containing mean_luminance, dark_pixel_ratio, is_low_light, low_light_score.
        """
        if frame is None or frame.size == 0:
            return {
                "mean_luminance": 0.0,
                "std_luminance": 0.0,
                "dark_pixel_ratio": 1.0,
                "is_low_light": True,
                "low_light_score": 1.0,
            }

        # Convert BGR to YCrCb and LAB for robust perceptual luminance calculation
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]

        mean_lum = float(np.mean(y_channel))
        std_lum = float(np.std(y_channel))

        # Ratio of dark pixels (below extreme_dark_threshold)
        dark_pixels = np.count_nonzero(y_channel < self.extreme_dark_threshold)
        total_pixels = y_channel.size
        dark_pixel_ratio = float(dark_pixels / total_pixels)

        # Low-light score normalized to [0, 1]
        # 0 = bright, 1 = severely dark
        score = np.clip(1.0 - (mean_lum / 120.0), 0.0, 1.0)
        # Boost score if a high fraction of pixels are pitch dark
        if dark_pixel_ratio > 0.3:
            score = float(np.clip(score + 0.3 * dark_pixel_ratio, 0.0, 1.0))

        is_low_light = bool(
            mean_lum < self.dark_luminance_threshold
            or dark_pixel_ratio > self.dark_pixel_ratio_threshold
        )

        return {
            "mean_luminance": round(mean_lum, 2),
            "std_luminance": round(std_lum, 2),
            "dark_pixel_ratio": round(dark_pixel_ratio, 4),
            "low_light_score": round(score, 3),
            "is_low_light": is_low_light,
        }
