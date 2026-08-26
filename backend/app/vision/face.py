import numpy as np
from typing import List, Dict, Any

class SCRFDFaceDetector:
    """SCRFD face detector for fast and accurate face detection in surveillance streams."""

    def __init__(self, model_path: str = "models/scrfd.onnx"):
        self.model_path = model_path

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detects faces in frame and returns bounding boxes + facial landmarks."""
        return []
