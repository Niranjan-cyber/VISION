"""
Resolution and spatial detail analyzer for CCTV footage.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple


class ResolutionAnalyzer:
    """
    Evaluates spatial dimensions, aspect ratio, and fine-edge detail density.
    """

    def __init__(
        self,
        min_target_width: int = 1280,
        min_target_height: int = 720,
        min_edge_density: float = 0.015,
    ):
        """
        Args:
            min_target_width: Minimum width for HD CCTV footage.
            min_target_height: Minimum height for HD CCTV footage.
            min_edge_density: Ratio of high-frequency edge pixels.
        """
        self.min_target_width = min_target_width
        self.min_target_height = min_target_height
        self.min_edge_density = min_edge_density

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze spatial resolution and detail density of a BGR frame.

        Returns:
            Dict containing width, height, megapixels, edge_density, is_low_res.
        """
        if frame is None or frame.size == 0:
            return {
                "width": 0,
                "height": 0,
                "megapixels": 0.0,
                "edge_density": 0.0,
                "is_low_res": True,
            }

        h, w = frame.shape[:2]
        mp = (w * h) / 1_000_000.0

        # Fine-edge detail density via Canny edge detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        edge_density = float(edge_pixels / (w * h))

        is_low_res = bool(w < self.min_target_width or h < self.min_target_height)

        return {
            "width": w,
            "height": h,
            "megapixels": round(mp, 2),
            "edge_density": round(edge_density, 4),
            "is_low_res": is_low_res,
        }
