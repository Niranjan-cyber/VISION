"""
Video writer module for exporting processed footage and split-screen comparison videos.
Supports H.264 web-compatible encoding via imageio-ffmpeg and OpenCV cv2.VideoWriter fallbacks.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, List

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


class VideoWriter:
    """
    Video writer wrapper supporting web-compatible H.264 encoding and split-screen HUD overlays.
    """

    def __init__(
        self,
        output_path: str,
        fps: float = 25.0,
        frame_size: Optional[Tuple[int, int]] = None,
        fourcc: str = "mp4v",
        prefer_h264: bool = True,
    ):
        """
        Initialize VideoWriter.

        Args:
            output_path: Output video file path (e.g., enhanced_videos/out.mp4).
            fps: Video frames per second.
            frame_size: (width, height) tuple. If None, initialized upon first frame write.
            fourcc: 4-character codec code ('mp4v', 'avc1', 'XVID').
            prefer_h264: Use imageio libx264 for universal HTML5 browser playback.
        """
        self.output_path = output_path
        self.fps = fps if fps > 0 else 25.0
        self.frame_size = frame_size
        self.fourcc_str = fourcc
        self.prefer_h264 = prefer_h264
        self.writer = None
        self.use_imageio = False
        self.frames_written = 0

        # Ensure parent directory exists
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        if self.frame_size is not None:
            self._init_writer(self.frame_size)

    def _init_writer(self, frame_size: Tuple[int, int]):
        self.frame_size = frame_size

        if self.prefer_h264 and HAS_IMAGEIO:
            try:
                self.writer = imageio.get_writer(
                    self.output_path,
                    fps=self.fps,
                    codec="libx264",
                    quality=8,
                    pixelformat="yuv420p",
                    ffmpeg_params=["-preset", "ultrafast", "-crf", "22"],
                )
                self.use_imageio = True
                return
            except Exception as e:
                self.use_imageio = False

        # Fallback to cv2.VideoWriter
        codecs_to_try = [self.fourcc_str, "mp4v", "avc1", "XVID", "MJPG"]
        for c in codecs_to_try:
            try:
                fourcc_code = cv2.VideoWriter_fourcc(*c)
                self.writer = cv2.VideoWriter(
                    self.output_path,
                    fourcc_code,
                    self.fps,
                    self.frame_size,
                    isColor=True,
                )
                if self.writer.isOpened():
                    self.fourcc_str = c
                    self.use_imageio = False
                    return
            except Exception:
                continue

        if self.writer is None or (not self.use_imageio and not self.writer.isOpened()):
            raise RuntimeError(f"Failed to initialize video writer for path {self.output_path}")

    def write_frame(self, frame: np.ndarray):
        """
        Write a single BGR frame.
        """
        if frame is None:
            return

        h, w = frame.shape[:2]
        if self.writer is None:
            self._init_writer((w, h))

        # Resize if dimensions do not match initial frame_size
        if (w, h) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if self.use_imageio:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.writer.append_data(rgb_frame)
        else:
            self.writer.write(frame)

        self.frames_written += 1

    def write_batch(self, frames: List[np.ndarray]):
        for f in frames:
            self.write_frame(f)

    @staticmethod
    def create_comparison_frame(
        original: np.ndarray,
        enhanced: np.ndarray,
        left_label: str = "Original CCTV",
        right_label: str = "Enhanced (SIH-2026)",
        metrics_text: Optional[str] = None,
    ) -> np.ndarray:
        """
        Horizontally concatenate original and enhanced frames with badges and HUD.
        """
        h_orig, w_orig = original.shape[:2]
        h_enh, w_enh = enhanced.shape[:2]
        target_h = max(240, int((min(max(h_orig, h_enh), 720) // 16) * 16))

        # Scale original to target_h
        scale_orig = target_h / max(1, h_orig)
        target_w_orig = max(16, int((int(w_orig * scale_orig) // 16) * 16))
        orig_resized = cv2.resize(original, (target_w_orig, target_h), interpolation=cv2.INTER_LINEAR)

        # Scale enhanced to target_h
        scale_enh = target_h / max(1, h_enh)
        target_w_enh = max(16, int((int(w_enh * scale_enh) // 16) * 16))
        enh_resized = cv2.resize(enhanced, (target_w_enh, target_h), interpolation=cv2.INTER_LINEAR)

        # Draw left label badge
        VideoWriter._draw_badge(orig_resized, left_label, bg_color=(40, 40, 40), text_color=(255, 255, 255))

        # Draw right label badge
        VideoWriter._draw_badge(enh_resized, right_label, bg_color=(20, 120, 20), text_color=(255, 255, 255))

        if metrics_text:
            VideoWriter._draw_metrics_bar(enh_resized, metrics_text)

        # Concatenate horizontally with a thin divider
        divider = np.full((target_h, 4, 3), 200, dtype=np.uint8)
        combined = np.hstack([orig_resized, divider, enh_resized])

        # Ensure total width is divisible by 16
        tot_h, tot_w = combined.shape[:2]
        pad_w = (16 - tot_w % 16) % 16
        if pad_w > 0:
            combined = cv2.copyMakeBorder(combined, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return combined

    @staticmethod
    def _draw_badge(img: np.ndarray, text: str, bg_color=(30, 30, 30), text_color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2
        margin = 10
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x1, y1 = margin, margin
        x2, y2 = margin + tw + 16, margin + th + 16
        cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(
            img,
            text,
            (x1 + 8, y1 + th + 6),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_metrics_bar(img: np.ndarray, text: str):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        h, w = img.shape[:2]
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        y1 = h - th - 20
        y2 = h - 6
        x1 = 10
        x2 = min(w - 10, x1 + tw + 16)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 1)
        cv2.putText(
            img,
            text,
            (x1 + 8, y2 - 6),
            font,
            font_scale,
            (0, 220, 255),
            thickness,
            cv2.LINE_AA,
        )

    def release(self):
        if self.use_imageio and self.writer is not None:
            self.writer.close()
        elif self.writer is not None and hasattr(self.writer, "isOpened") and self.writer.isOpened():
            self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
