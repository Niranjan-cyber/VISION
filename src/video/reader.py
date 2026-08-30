"""
Video reader module supporting local video files, image sequences, and RTSP / IP streams.
Includes optional background threaded prefetching for high-FPS decoding.
"""

import os
import cv2
import queue
import threading
from typing import Generator, Tuple, Optional, List
import numpy as np


class VideoReader:
    """
    High-performance video reader using OpenCV cv2.VideoCapture.
    Supports file reading, RTSP / IP camera streams, and frame seeking.
    """

    def __init__(self, source: str, queue_size: int = 64, use_threading: bool = False):
        """
        Initialize VideoReader.
        
        Args:
            source: Path to video file or RTSP stream URL or camera index.
            queue_size: Buffer queue size for threaded reading.
            use_threading: Whether to prefetch frames in a background thread.
        """
        self.source = source
        self.use_threading = use_threading
        self.queue_size = queue_size

        # Resolve camera index if string digit
        if isinstance(source, str) and source.isdigit():
            self._src = int(source)
        else:
            self._src = source

        self.cap = cv2.VideoCapture(self._src)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")

        # Metadata
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 25.0  # Default CCTV fps
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames < 0:
            self.total_frames = 0
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0.0

        # Threading state
        self._queue = None
        self._thread = None
        self._stopped = False

        if self.use_threading:
            self._start_worker()

    def _start_worker(self):
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._stopped = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stopped:
            if not self._queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self._queue.put((False, None))
                    break
                self._queue.put((True, frame))
            else:
                cv2.waitKey(1)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the video source.
        Returns (success_flag, frame_bgr).
        """
        if self.use_threading and self._queue is not None:
            try:
                ret, frame = self._queue.get(timeout=2.0)
                return ret, frame
            except queue.Empty:
                return False, None
        else:
            ret, frame = self.cap.read()
            return ret, frame

    def get_frame_at(self, frame_index: int) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Seek and retrieve a specific frame by index (file sources only).
        """
        if self.use_threading:
            raise RuntimeError("Direct frame seeking is not supported in threaded mode.")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        return self.cap.read()

    def iter_frames(self, max_frames: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yield (frame_index, frame_bgr) generator.
        """
        count = 0
        while True:
            if max_frames is not None and count >= max_frames:
                break
            ret, frame = self.read_frame()
            if not ret or frame is None:
                break
            yield count, frame
            count += 1

    def read_batch(self, batch_size: int) -> List[np.ndarray]:
        """
        Read a batch of frames as a list of numpy arrays.
        """
        batch = []
        for _ in range(batch_size):
            ret, frame = self.read_frame()
            if not ret or frame is None:
                break
            batch.append(frame)
        return batch

    def reset(self):
        """Rewind back to the start of the video."""
        if self.use_threading:
            self.release()
            self.cap = cv2.VideoCapture(self._src)
            self._start_worker()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        """Release video capture resources."""
        self._stopped = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __iter__(self):
        for _, frame in self.iter_frames():
            yield frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def get_info(self) -> dict:
        """Return dictionary summary of video properties."""
        return {
            "source": str(self.source),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration_sec, 2),
        }
