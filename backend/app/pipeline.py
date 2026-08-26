import logging
from app.ingestion.source import VideoSource
from app.vision.detector import YOLODetector
from app.vision.tracker import ByteTrackerHandler
from app.events.virtual_fence import VirtualFence
from app.events.engine import RiskEngine

logger = logging.getLogger(__name__)

class ProcessingPipeline:
    """Master Vision Processing Pipeline uniting Stream Ingestion, AI Detection, Tracking, and Risk Alerts."""

    def __init__(self, camera_id: str, stream_url: str):
        self.camera_id = camera_id
        self.source = VideoSource(stream_url)
        self.detector = YOLODetector()
        self.tracker = ByteTrackerHandler()
        self.risk_engine = RiskEngine()

    def process_frame(self, frame):
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)
        return tracks
