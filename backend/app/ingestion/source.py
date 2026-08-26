import cv2
import logging

logger = logging.getLogger(__name__)

class VideoSource:
    """Handles RTSP/IP camera feed or local video file frame ingestion via OpenCV/FFmpeg."""

    def __init__(self, source_url: str):
        self.source_url = source_url
        self.cap = None

    def connect(self) -> bool:
        self.cap = cv2.VideoCapture(self.source_url)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source_url}")
            return False
        logger.info(f"Successfully connected to video source: {self.source_url}")
        return True

    def read_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def release(self):
        if self.cap:
            self.cap.release()
            logger.info(f"Released video source: {self.source_url}")
