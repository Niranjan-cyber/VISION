import os
import sys
from typing import List, Optional
import cv2
import numpy as np

from src.core.types import BoundingBox, PlateDetection


class LicensePlateDetector:
    """
    License Plate Detector for vehicle crops.
    Supports a deep learning model (YOLO) with an OpenCV morphological
    edge/aspect-ratio fallback engine.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.40,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.yolo_model = None

        if self.model_path and os.path.exists(self.model_path):
            self._load_yolo_model()

    def _load_yolo_model(self) -> None:
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.model_path)
        except Exception as e:
            print(
                f"[WARNING] Failed to load custom YOLO plate model '{self.model_path}': {e}. "
                "Using morphological edge detector.",
                file=sys.stderr,
            )
            self.yolo_model = None

    def detect(self, vehicle_crop: np.ndarray) -> List[PlateDetection]:
        """
        Detects license plates in a vehicle crop.

        Args:
            vehicle_crop: np.ndarray BGR vehicle image crop.

        Returns:
            List of PlateDetection objects with coordinates relative to the vehicle crop.
        """
        if vehicle_crop is None or not isinstance(vehicle_crop, np.ndarray) or vehicle_crop.size == 0:
            return []

        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 20:
            return []

        # 1. Use deep YOLO model if available
        if self.yolo_model is not None:
            detections = self._detect_yolo(vehicle_crop)
            if detections:
                return detections

        # 2. Morphological / Contour fallback
        return self._detect_morphological(vehicle_crop)

    def _detect_yolo(self, vehicle_crop: np.ndarray) -> List[PlateDetection]:
        try:
            results = self.yolo_model(vehicle_crop, conf=self.confidence_threshold, verbose=False)
            detections: List[PlateDetection] = []
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    bbox = BoundingBox(
                        x1=int(xyxy[0]),
                        y1=int(xyxy[1]),
                        x2=int(xyxy[2]),
                        y2=int(xyxy[3]),
                    )
                    detections.append(PlateDetection(bbox=bbox, confidence=conf))
            return detections
        except Exception as e:
            print(f"[WARNING] YOLO plate detection failed: {e}", file=sys.stderr)
            return []

    def _detect_morphological(self, vehicle_crop: np.ndarray) -> List[PlateDetection]:
        """
        Extracts license plate candidates using vertical edge density,
        morphological closure, and aspect ratio filtering.
        """
        h, w = vehicle_crop.shape[:2]
        vehicle_area = h * w
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)

        # A. Bilateral blur to reduce noise while preserving edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)

        # B. Sobel vertical edge filter (license plates have dense vertical character edges)
        grad_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)

        # C. Morphological closing to connect adjacent vertical characters into a single blob
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(abs_grad_x, cv2.MORPH_CLOSE, kernel)

        # D. Otsu thresholding
        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # E. Clean up noise
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)

        # F. Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[PlateDetection] = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Minimum readable dimensions: plates cannot physically be smaller than 45x12
            if ch < 12 or cw < 45:
                continue

            area = cw * ch
            aspect_ratio = cw / float(ch)

            # Aspect ratio for standard Indian/International plates is ~2.0 to 5.5
            if not (1.8 <= aspect_ratio <= 6.0):
                continue

            # Plate must occupy between 0.3% and 30% of the vehicle crop
            area_ratio = area / float(vehicle_area)
            if not (0.003 <= area_ratio <= 0.30):
                continue

            # License plates are located in the lower 70% of the vehicle
            if (y + ch) < (0.30 * h):
                continue

            # Rectangularity check
            extent = cv2.contourArea(cnt) / float(area)
            if extent < 0.35:
                continue

            # Confidence estimated based on aspect ratio proximity to ideal (3.8), extent, and readable size
            ideal_ar = 3.8
            ar_score = max(0.0, 1.0 - abs(aspect_ratio - ideal_ar) / 3.0)
            size_bonus = 0.15 if cw >= 75 else 0.0
            conf = min(0.95, max(0.40, 0.45 * ar_score + 0.40 * extent + size_bonus))

            bbox = BoundingBox(x1=x, y1=y, x2=x + cw, y2=y + ch)
            candidates.append(PlateDetection(bbox=bbox, confidence=float(conf)))

        # Sort candidates by confidence descending
        candidates.sort(key=lambda d: d.confidence, reverse=True)
        return candidates[:3]  # Return top candidate(s)
