"""
Contrast and dynamic range analyzer for CCTV video frames.
"""

import cv2
import numpy as np
from typing import Dict, Any


class ContrastAnalyzer:
    """
    Evaluates Root-Mean-Square (RMS) contrast, Michelson contrast, and dynamic range.
    """

    def __init__(
        self,
        rms_threshold: float = 0.15,
        dynamic_range_threshold: float = 80.0,
    ):
        """
        Args:
            rms_threshold: RMS contrast threshold below which image is low-contrast.
            dynamic_range_threshold: Intensity span (p99 - p1) threshold.
        """
        self.rms_threshold = rms_threshold
        self.dynamic_range_threshold = dynamic_range_threshold

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze contrast properties of a BGR frame.

        Returns:
            Dict with rms_contrast, michelson_contrast, dynamic_range, is_low_contrast.
        """
        if frame is None or frame.size == 0:
            return {
                "rms_contrast": 0.0,
                "michelson_contrast": 0.0,
                "dynamic_range": 0.0,
                "is_low_contrast": True,
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        norm_gray = gray.astype(np.float32) / 255.0

        # 1. RMS Contrast
        rms = float(np.std(norm_gray))

        # 2. Michelson Contrast
        i_min = float(np.min(gray))
        i_max = float(np.max(gray))
        michelson = float((i_max - i_min) / (i_max + i_min + 1e-6))

        # 3. Dynamic Range (99th percentile - 1st percentile)
        p1, p99 = np.percentile(gray, [1, 99])
        dynamic_range = float(p99 - p1)

        is_low_contrast = bool(
            rms < self.rms_threshold or dynamic_range < self.dynamic_range_threshold
        )

        return {
            "rms_contrast": round(rms, 4),
            "michelson_contrast": round(michelson, 4),
            "dynamic_range": round(dynamic_range, 2),
            "is_low_contrast": is_low_contrast,
        }
