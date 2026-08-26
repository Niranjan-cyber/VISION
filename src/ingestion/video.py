import os
from typing import Optional, Tuple
import cv2
import numpy as np


class VideoSource:
    """Reusable OpenCV video source abstraction for file ingestion."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._width: int = 0
        self._height: int = 0
        self._frame_count: int = 0
        self._current_frame: int = 0

        self._open_source()

    def _open_source(self) -> None:
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

    def read_frame(self) -> Optional[np.ndarray]:
        """Reads the next frame sequentially. Returns None if end of video or error."""
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
