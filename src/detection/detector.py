import sys
from typing import List, Optional, Set
import numpy as np

from src.core.types import BoundingBox, Detection


class YOLODetector:
    """YOLO11n object detector for VISION surveillance target classes."""

    TARGET_CLASSES: Set[str] = {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
    }

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads the Ultralytics YOLO model with clear error handling."""
        try:
            from ultralytics import YOLO
        except ImportError:
            print(
                "[ERROR] 'ultralytics' package is not installed. "
                "Please run: pip install ultralytics",
                file=sys.stderr,
            )
            raise

        try:
            self.model = YOLO(self.model_name)
        except Exception as e:
            print(
                f"[ERROR] Failed to load YOLO model '{self.model_name}': {e}",
                file=sys.stderr,
            )
            raise RuntimeError(
                f"Unable to initialize YOLODetector with model '{self.model_name}': {e}"
            ) from e

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """
        Runs object detection on the input frame and converts Ultralytics outputs
        into a list of VISION Detection domain objects.
        """
        if self.model is None or frame is None:
            return []

        threshold = (
            conf_threshold
            if conf_threshold is not None
            else self.confidence_threshold
        )

        results = self.model.predict(
            source=frame,
            conf=threshold,
            verbose=False,
        )

        detections: List[Detection] = []

        if not results:
            return detections

        first_result = results[0]
        boxes = first_result.boxes
        if boxes is None:
            return detections

        names = first_result.names

        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = names.get(cls_id, f"class_{cls_id}").lower()

            if class_name not in self.TARGET_CLASSES:
                continue

            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            detection = Detection(
                class_id=cls_id,
                class_name=class_name,
                confidence=conf,
                bbox=bbox,
            )
            detections.append(detection)

        return detections
