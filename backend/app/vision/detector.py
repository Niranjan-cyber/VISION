from typing import List, Dict, Any
import numpy as np

class YOLODetector:
    """YOLO model wrapper for detecting Person, Car, Truck, Bus, Motorcycle, Bicycle."""

    def __init__(self, model_path: str = "models/yolov8n.pt", confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]

    def load_model(self):
        """Loads Ultralytics YOLO model."""
        pass

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on input frame.
        Returns list of detections with bbox [x1, y1, x2, y2], confidence, class_name.
        """
        return []
