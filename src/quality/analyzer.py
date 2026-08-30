"""
Comprehensive Quality Analyzer module for CCTV footage.
Assesses brightness, blur, contrast, and resolution to determine quality status
and suggest targeted restoration pipelines.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .brightness import BrightnessAnalyzer
from .blur import BlurAnalyzer
from .contrast import ContrastAnalyzer
from .resolution import ResolutionAnalyzer


@dataclass
class QualityReport:
    """Detailed quality diagnostic report for a single frame or video segment."""
    is_good_quality: bool
    is_low_light: bool
    is_blurry: bool
    is_low_res: bool
    is_low_contrast: bool
    brightness: Dict[str, Any]
    blur: Dict[str, Any]
    contrast: Dict[str, Any]
    resolution: Dict[str, Any]
    recommended_enhancements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": "GOOD" if self.is_good_quality else "POOR",
            "is_good_quality": self.is_good_quality,
            "is_low_light": self.is_low_light,
            "is_blurry": self.is_blurry,
            "is_low_res": self.is_low_res,
            "is_low_contrast": self.is_low_contrast,
            "recommended_enhancements": self.recommended_enhancements,
            "brightness": self.brightness,
            "blur": self.blur,
            "contrast": self.contrast,
            "resolution": self.resolution,
        }

    def get_hud_string(self) -> str:
        """Compact text string for live video HUD display."""
        actions = "+".join(self.recommended_enhancements) if self.recommended_enhancements else "PASSTHROUGH"
        lum = self.brightness.get("mean_luminance", 0)
        lap = self.blur.get("laplacian_var", 0)
        w = self.resolution.get("width", 0)
        h = self.resolution.get("height", 0)
        status = "GOOD" if self.is_good_quality else "POOR"
        return f"Status: {status} | Lum: {lum:.1f} | Sharp: {lap:.1f} | Res: {w}x{h} | Pipe: {actions}"


class QualityAnalyzer:
    """
    Master video quality analyzer combining individual metric extractors.
    """

    def __init__(
        self,
        brightness_threshold: float = 60.0,
        blur_threshold: float = 120.0,
        min_width: int = 1280,
        min_height: int = 720,
    ):
        self.brightness_analyzer = BrightnessAnalyzer(dark_luminance_threshold=brightness_threshold)
        self.blur_analyzer = BlurAnalyzer(laplacian_threshold=blur_threshold)
        self.contrast_analyzer = ContrastAnalyzer()
        self.resolution_analyzer = ResolutionAnalyzer(min_target_width=min_width, min_target_height=min_height)

    def analyze_frame(self, frame: np.ndarray) -> QualityReport:
        """
        Perform multi-metric quality evaluation on a single BGR frame.
        """
        if frame is None or frame.size == 0:
            return QualityReport(
                is_good_quality=False,
                is_low_light=True,
                is_blurry=True,
                is_low_res=True,
                is_low_contrast=True,
                brightness={"mean_luminance": 0, "is_low_light": True},
                blur={"laplacian_var": 0, "is_blurry": True},
                contrast={"rms_contrast": 0, "is_low_contrast": True},
                resolution={"width": 0, "height": 0, "is_low_res": True},
                recommended_enhancements=["zero_dce", "rvrt", "realesrgan"],
            )

        b_res = self.brightness_analyzer.analyze(frame)
        bl_res = self.blur_analyzer.analyze(frame)
        c_res = self.contrast_analyzer.analyze(frame)
        r_res = self.resolution_analyzer.analyze(frame)

        is_low_light = b_res["is_low_light"]
        is_blurry = bl_res["is_blurry"]
        is_low_res = r_res["is_low_res"]
        is_low_contrast = c_res["is_low_contrast"]

        recommendations = []
        if is_low_light:
            recommendations.append("zero_dce")
        if is_blurry:
            recommendations.append("rvrt")
        if is_low_res:
            recommendations.append("realesrgan")

        is_good = not (is_low_light or is_blurry or is_low_res or is_low_contrast)

        return QualityReport(
            is_good_quality=is_good,
            is_low_light=is_low_light,
            is_blurry=is_blurry,
            is_low_res=is_low_res,
            is_low_contrast=is_low_contrast,
            brightness=b_res,
            blur=bl_res,
            contrast=c_res,
            resolution=r_res,
            recommended_enhancements=recommendations,
        )

    def analyze_batch(self, frames: List[np.ndarray]) -> List[QualityReport]:
        return [self.analyze_frame(f) for f in frames]
