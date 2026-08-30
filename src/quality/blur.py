"""
Blur and motion degradation analyzer for CCTV surveillance frames.
Combines spatial Laplacian variance, Tenengrad gradient energy, and Frequency Domain (FFT) analysis.
"""

import cv2
import numpy as np
from typing import Dict, Any


class BlurAnalyzer:
    """
    Analyzes sharpness, edge energy, and frequency spectrum to detect CCTV motion blur and defocus.
    """

    def __init__(
        self,
        laplacian_threshold: float = 120.0,
        fft_threshold: float = 0.08,
    ):
        """
        Args:
            laplacian_threshold: Variance of Laplacian below which frame is flagged as blurry.
            fft_threshold: Ratio of high-frequency power below which frame is blurry.
        """
        self.laplacian_threshold = laplacian_threshold
        self.fft_threshold = fft_threshold

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze blur and sharpness metrics of a BGR frame.

        Returns:
            Dict containing laplacian_var, tenengrad_energy, fft_hf_ratio, blur_score, is_blurry.
        """
        if frame is None or frame.size == 0:
            return {
                "laplacian_var": 0.0,
                "tenengrad_energy": 0.0,
                "fft_hf_ratio": 0.0,
                "blur_score": 1.0,
                "is_blurry": True,
            }

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Variance of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(laplacian.var())

        # 2. Tenengrad Gradient Energy (Sobel x & y squared)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag_sq = gx ** 2 + gy ** 2
        tenengrad = float(np.mean(grad_mag_sq))

        # 3. FFT High-Frequency Energy Ratio (evaluated on normalized canvas for speed)
        h, w = gray.shape
        gray_fft = cv2.resize(gray, (640, 360)) if (h > 720 or w > 1280) else gray
        fh, fw = gray_fft.shape
        cy, cx = fh // 2, fw // 2
        f = np.fft.fft2(gray_fft)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

        # High-pass mask (zero out central low frequency radius)
        radius = min(fh, fw) // 8
        y, x = np.ogrid[:fh, :fw]
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        high_freq_mask = dist_from_center > radius

        total_energy = np.sum(magnitude_spectrum) + 1e-8
        hf_energy = np.sum(magnitude_spectrum[high_freq_mask])
        fft_hf_ratio = float(hf_energy / total_energy)

        # Calculate normalized blur score in [0, 1]
        # 0 = razor sharp, 1 = severely blurred
        # Map lap_var: typical sharp images have lap_var > 250; blurry < 100
        sharpness_factor = min(lap_var / 300.0, 1.0)
        blur_score = float(np.clip(1.0 - sharpness_factor, 0.0, 1.0))

        is_blurry = bool(lap_var < self.laplacian_threshold or fft_hf_ratio < self.fft_threshold)

        return {
            "laplacian_var": round(lap_var, 2),
            "tenengrad_energy": round(tenengrad, 2),
            "fft_hf_ratio": round(fft_hf_ratio, 4),
            "blur_score": round(blur_score, 3),
            "is_blurry": is_blurry,
        }
