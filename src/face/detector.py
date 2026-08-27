import os
import sys
import urllib.request
from typing import List, Optional
import cv2
import numpy as np

from src.core.types import BoundingBox, FaceDetection

YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


class FaceDetector:
    """Pretrained deep learning face detector (YuNet / SCRFD) operating on image crops."""

    def __init__(
        self,
        model_path: str = "models/face_detection_yunet_2023mar.onnx",
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ):
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.detector = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads YuNet ONNX face detection model, downloading if necessary."""
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)

        if not os.path.exists(self.model_path):
            print(f"[INFO] Face model not found locally at '{self.model_path}'. Downloading...", file=sys.stderr)
            try:
                urllib.request.urlretrieve(YUNET_MODEL_URL, self.model_path)
                print(f"[INFO] Successfully downloaded face model to '{self.model_path}'.", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] Failed to download face model: {e}", file=sys.stderr)
                raise RuntimeError(f"Unable to download face model from {YUNET_MODEL_URL}") from e

        try:
            # Initialize OpenCV FaceDetectorYN
            self.detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (300, 300),
                self.score_threshold,
                self.nms_threshold,
                5000,
            )
        except Exception as e:
            print(f"[ERROR] Failed to initialize FaceDetectorYN: {e}", file=sys.stderr)
            raise RuntimeError(f"FaceDetector initialization error: {e}") from e

    def detect(self, crop: np.ndarray) -> List[FaceDetection]:
        """
        Detects faces in the given image or person crop.
        Returns crop-relative FaceDetection objects.
        """
        if self.detector is None or crop is None or crop.size == 0:
            return []

        h, w = crop.shape[:2]
        if h < 5 or w < 5:
            return []

        try:
            self.detector.setInputSize((w, h))
            status, faces = self.detector.detect(crop)
        except Exception as e:
            print(f"[WARNING] Face detection failed on crop ({w}x{h}): {e}", file=sys.stderr)
            return []

        if status != 1 or faces is None or len(faces) == 0:
            return []

        face_detections: List[FaceDetection] = []
        for face in faces:
            # YuNet output: [x, y, w, h, x_re, y_re, x_le, y_le, x_n, y_n, x_rm, y_rm, x_lm, y_lm, score]
            fx, fy, fw, fh = float(face[0]), float(face[1]), float(face[2]), float(face[3])
            score = float(face[14])

            x1 = max(0, int(round(fx)))
            y1 = max(0, int(round(fy)))
            x2 = min(w, int(round(fx + fw)))
            y2 = min(h, int(round(fy + fh)))

            if x2 <= x1 or y2 <= y1:
                continue

            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            face_detections.append(FaceDetection(bbox=bbox, confidence=score))

        return face_detections
