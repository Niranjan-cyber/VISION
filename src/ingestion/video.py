import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np


class VideoSource:
    """Reusable OpenCV video source abstraction. Wraps either a recorded file
    (VideoCapture(path)) or a live local camera device (VideoCapture(index)) —
    exactly one of video_path / device_index must be given. Callers can tell
    the two apart via `is_live` without needing to know which one was passed."""

    def __init__(self, video_path: Optional[str] = None, device_index: Optional[int] = None):
        if (video_path is None) == (device_index is None):
            raise ValueError("VideoSource requires exactly one of video_path or device_index")

        self.video_path = video_path
        self.device_index = device_index
        self.is_live = device_index is not None
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._width: int = 0
        self._height: int = 0
        self._frame_count: int = 0
        self._current_frame: int = 0

        if self.is_live:
            self._open_device()
        else:
            self._open_file()

    def _open_file(self) -> None:
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(
                f"Video file not found at path: '{self.video_path}'"
            )

        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise ValueError(
                f"Unable to open video source: '{self.video_path}'"
            )

        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if self._fps <= 0:
            self._fps = 30.0  # Fallback default if FPS unavailable from header

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame = 0

    def _open_device(self) -> None:
        # DirectShow opens local USB/webcam devices far more reliably (and
        # faster) than the default MSMF backend on Windows; other platforms
        # use OpenCV's default backend selection.
        if sys.platform == "win32":
            cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.device_index)
        else:
            cap = cv2.VideoCapture(self.device_index)

        if not cap.isOpened():
            raise ValueError(
                f"Unable to open camera device index {self.device_index}. "
                "It may not exist, may already be in use by another application "
                "or camera stream, or may not be accessible."
            )

        self._cap = cap
        # A live device frequently misreports FPS/resolution until the first
        # frame is actually grabbed — 0 is a legitimate transient value, not
        # an error, and read_frame() below is what actually proves the device
        # works.
        reported_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._fps = reported_fps if reported_fps > 0 else 30.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = 0  # unknown/unbounded for a live source
        self._current_frame = 0

    def read_frame(self) -> Optional[np.ndarray]:
        """Reads the next frame. Returns None at end-of-file for a recorded
        source, or on a read failure (e.g. device disconnected) for a live
        source — callers must treat those two None cases differently."""
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        self._current_frame += 1
        return frame

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def dimensions(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def release(self) -> None:
        """Releases the OpenCV capture cleanly."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def discover_camera_devices(max_index: int = 5) -> List[dict]:
    """Probes local camera device indices 0..max_index-1 by actually
    attempting to open (and read one frame from) each one — never reports a
    device as available just because an index number exists. A device
    already claimed by an active CameraSession, or that doesn't exist, comes
    back as unavailable rather than guessed-at.

    Returns a list of {"device_index": int, "available": bool, "width": int,
    "height": int} — width/height are 0 when unavailable.
    """
    results = []
    for idx in range(max_index):
        entry = {"device_index": idx, "available": False, "width": 0, "height": 0}
        try:
            if sys.platform == "win32":
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    entry["available"] = True
                    entry["height"], entry["width"] = frame.shape[:2]
            cap.release()
        except Exception:
            entry["available"] = False
        results.append(entry)
    return results
