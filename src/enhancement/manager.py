"""
Unified Enhancement Manager for SIH 2026 CCTV Intelligent Video Enhancement.
Coordinates quality diagnostics and dynamically routes degraded frames to specialized
deep learning models (Zero-DCE++, Real-ESRGAN, BasicVSR++, RVRT).
"""

import os
import time
import cv2
import numpy as np
import torch
from typing import Optional, List, Tuple, Dict, Any
from tqdm import tqdm

from ..quality.analyzer import QualityAnalyzer, QualityReport
from ..video.reader import VideoReader
from ..video.writer import VideoWriter
from .low_light.zero_dce import ZeroDCEEnhancer
from .low_resolution.realesrgan import RealESRGANEnhancer
from .low_resolution.basicvsr import BasicVSREnhancer
from .blur.rvrt import RVRTEnhancer


class EnhancementManager:
    """
    Master pipeline orchestrator that analyzes incoming CCTV frames and dispatches
    targeted deep learning enhancement algorithms based on degradation signatures.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        enable_zero_dce: bool = True,
        enable_realesrgan: bool = True,
        enable_basicvsr: bool = True,
        enable_rvrt: bool = True,
        auto_download_weights: bool = True,
        quality_analyzer: Optional[QualityAnalyzer] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[EnhancementManager] Initializing pipeline on device: {self.device}")

        # Quality Analyzer
        self.quality_analyzer = quality_analyzer or QualityAnalyzer()

        # Enhancer models
        self.zero_dce = ZeroDCEEnhancer(device=self.device, auto_download=auto_download_weights) if enable_zero_dce else None
        self.realesrgan = RealESRGANEnhancer(device=self.device, auto_download=auto_download_weights) if enable_realesrgan else None
        self.basicvsr = BasicVSREnhancer(device=self.device) if enable_basicvsr else None
        self.rvrt = RVRTEnhancer(device=self.device) if enable_rvrt else None

    def process_frame(
        self,
        frame: np.ndarray,
        mode: str = "auto",
        force_enhancers: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, QualityReport]:
        """
        Process a single CCTV frame.

        Args:
            frame: Input BGR frame.
            mode: 'auto' (quality-driven routing), 'zero_dce', 'realesrgan', 'rvrt', 'basicvsr', or 'all'.
            force_enhancers: Optional list of enhancers to apply sequentially (e.g. ['zero_dce', 'rvrt']).

        Returns:
            Tuple of (enhanced_frame, QualityReport).
        """
        if frame is None or frame.size == 0:
            return frame, self.quality_analyzer.analyze_frame(frame)

        report = self.quality_analyzer.analyze_frame(frame)
        enhanced = frame.copy()

        if mode == "auto":
            applied = []
            # Step 1: Low-Light Enhancement
            if report.is_low_light and self.zero_dce is not None:
                enhanced = self.zero_dce.enhance(enhanced)
                applied.append("zero_dce")

            # Step 2: Motion/Defocus Deblurring
            if report.is_blurry and self.rvrt is not None:
                enhanced = self.rvrt.enhance(enhanced)
                applied.append("rvrt")

            # Step 3: Super-Resolution (if low-res)
            if report.is_low_res and self.realesrgan is not None:
                enhanced = self.realesrgan.enhance(enhanced)
                applied.append("realesrgan")

        elif mode == "zero_dce" and self.zero_dce is not None:
            enhanced = self.zero_dce.enhance(enhanced)
        elif mode == "realesrgan" and self.realesrgan is not None:
            enhanced = self.realesrgan.enhance(enhanced)
        elif mode == "rvrt" and self.rvrt is not None:
            enhanced = self.rvrt.enhance(enhanced)
        elif mode == "basicvsr" and self.basicvsr is not None:
            enhanced = self.basicvsr.enhance(enhanced)
        elif mode == "all":
            if self.zero_dce is not None:
                enhanced = self.zero_dce.enhance(enhanced)
            if self.rvrt is not None:
                enhanced = self.rvrt.enhance(enhanced)
            if self.realesrgan is not None:
                enhanced = self.realesrgan.enhance(enhanced)

        if force_enhancers:
            for enh_name in force_enhancers:
                if enh_name == "zero_dce" and self.zero_dce is not None:
                    enhanced = self.zero_dce.enhance(enhanced)
                elif enh_name == "rvrt" and self.rvrt is not None:
                    enhanced = self.rvrt.enhance(enhanced)
                elif enh_name == "realesrgan" and self.realesrgan is not None:
                    enhanced = self.realesrgan.enhance(enhanced)
        # Universal License Plate & Text Sharpness + Contrast Optimization (applied to every video)
        enhanced = self.enhance_text_and_plate_clarity(enhanced)

        return enhanced, report

    @staticmethod
    def enhance_text_and_plate_clarity(frame: np.ndarray) -> np.ndarray:
        """
        Universal high-clarity license plate & text detail booster.
        Applies luminance-only CLAHE contrast stretching, character stroke unsharp sharpening,
        and morphological alphanumeric ridge enhancement to make license plates and text readable.
        """
        if frame is None or frame.size == 0:
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Bilateral edge-preserving smoothing to eliminate compression noise
        smooth = cv2.bilateralFilter(l, d=5, sigmaColor=15, sigmaSpace=15)

        # 1. Local dynamic range CLAHE contrast boost on luminance
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l_clahe = clahe.apply(smooth)

        # 2. High-pass character stroke sharpening
        blur_l = cv2.GaussianBlur(l_clahe, (0, 0), sigmaX=1.0)
        high_pass = cv2.subtract(l_clahe, blur_l)
        l_sharp = cv2.addWeighted(l_clahe, 1.0, high_pass, 1.15, 0)

        # 3. Morphological Top-Hat & Black-Hat alphanumeric character edge enhancement
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        tophat = cv2.morphologyEx(l_clahe, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(l_clahe, cv2.MORPH_BLACKHAT, kernel)
        l_boost = cv2.add(l_sharp, cv2.multiply(tophat, 0.2, dtype=cv2.CV_8U))
        l_boost = cv2.subtract(l_boost, cv2.multiply(blackhat, 0.2, dtype=cv2.CV_8U))

        merged_lab = cv2.merge([l_boost, a, b])
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    def process_video(
        self,
        input_path: str,
        output_path: str,
        mode: str = "auto",
        create_comparison: bool = True,
        comparison_output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        step: int = 1,
        show_progress: bool = True,
        save_frames_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Enhance a full CCTV video file or stream, saving the enhanced video and optional comparison.

        Args:
            input_path: Source video path.
            output_path: Destination enhanced video path.
            mode: 'auto', 'zero_dce', 'realesrgan', 'rvrt', 'basicvsr', 'all'.
            create_comparison: Generate side-by-side split screen video.
            comparison_output_path: Destination for comparison video.
            max_frames: Max frames to process.
            step: Step interval for frame skipping (1 = every frame).
            show_progress: Display tqdm bar.
            save_frames_dir: Optional directory to save each enhanced frame as an image.

        Returns:
            Dict summarizing processing runtime, FPS, and quality metrics.
        """
        reader = VideoReader(input_path)
        total_frames = reader.total_frames
        fps = reader.fps
        if max_frames is not None:
            total_frames = min(total_frames, max_frames * step)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if comparison_output_path is None and create_comparison:
            base, ext = os.path.splitext(output_path)
            comparison_output_path = f"{base}_comparison{ext}"

        if save_frames_dir:
            os.makedirs(save_frames_dir, exist_ok=True)

        video_writer = None
        comp_writer = None
        if create_comparison and comparison_output_path:
            os.makedirs(os.path.dirname(comparison_output_path) or ".", exist_ok=True)

        start_time = time.time()
        processed_count = 0
        quality_counts = {"GOOD": 0, "LOW_LIGHT": 0, "BLURRY": 0, "LOW_RES": 0}

        pbar = tqdm(total=total_frames, desc=f"Enhancing {os.path.basename(input_path)}", disable=not show_progress)

        for frame_idx, orig_frame in reader.iter_frames(max_frames=max_frames * step if max_frames else None):
            if frame_idx % step != 0:
                pbar.update(1)
                continue

            enhanced_frame, report = self.process_frame(orig_frame, mode=mode)

            # Optionally save enhanced frame image
            if save_frames_dir:
                frame_filename = f"frame_{frame_idx:06d}_enhanced.png"
                cv2.imwrite(os.path.join(save_frames_dir, frame_filename), enhanced_frame)
            
            # Tally metrics
            if report.is_good_quality:
                quality_counts["GOOD"] += 1
            if report.is_low_light:
                quality_counts["LOW_LIGHT"] += 1
            if report.is_blurry:
                quality_counts["BLURRY"] += 1
            if report.is_low_res:
                quality_counts["LOW_RES"] += 1

            # Initialize writers on first frame
            if video_writer is None:
                eh, ew = enhanced_frame.shape[:2]
                video_writer = VideoWriter(output_path, fps=fps / step, frame_size=(ew, eh))

            video_writer.write_frame(enhanced_frame)

            if create_comparison:
                comp_frame = VideoWriter.create_comparison_frame(
                    orig_frame,
                    enhanced_frame,
                    left_label="RAW CCTV",
                    right_label=f"ENHANCED ({mode.upper()})",
                    metrics_text=report.get_hud_string(),
                )
                if comp_writer is None:
                    ch, cw = comp_frame.shape[:2]
                    comp_writer = VideoWriter(comparison_output_path, fps=fps / step, frame_size=(cw, ch))
                comp_writer.write_frame(comp_frame)

            processed_count += 1
            pbar.update(1)

        pbar.close()
        reader.release()
        if video_writer:
            video_writer.release()
        if comp_writer:
            comp_writer.release()

        elapsed = max(time.time() - start_time, 1e-4)
        proc_fps = processed_count / elapsed

        return {
            "input_video": input_path,
            "output_video": output_path,
            "comparison_video": comparison_output_path if create_comparison else None,
            "frames_processed": processed_count,
            "elapsed_seconds": round(elapsed, 2),
            "processing_fps": round(proc_fps, 2),
            "quality_distribution": quality_counts,
        }
