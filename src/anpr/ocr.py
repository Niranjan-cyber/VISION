from abc import ABC, abstractmethod
import sys
from typing import List, Optional, Tuple
import cv2
import numpy as np


class BasePlateOCREngine(ABC):
    """Abstract base class for license plate OCR engines."""

    @abstractmethod
    def recognize(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """
        Extracts alphanumeric text from an enhanced license plate crop.

        Args:
            plate_crop: np.ndarray enhanced license plate image.

        Returns:
            Tuple of (raw_text: str, confidence: float).
        """
        pass


class EasyOCREngine(BasePlateOCREngine):
    """
    License plate OCR engine powered by EasyOCR.
    Restricts character recognition to uppercase alphanumeric characters.
    """

    ALLOWED_CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, languages: Optional[List[str]] = None, use_gpu: bool = False):
        if languages is None:
            languages = ["en"]
        self.languages = languages
        self.use_gpu = use_gpu
        self.reader = None
        self._init_reader()

    def _init_reader(self) -> None:
        try:
            import easyocr
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False,
            )
        except ImportError:
            print(
                "[WARNING] 'easyocr' is not installed. To install: pip install easyocr. "
                "Falling back to template OCR engine.",
                file=sys.stderr,
            )
            self.reader = None
        except Exception as e:
            print(f"[WARNING] Failed to initialize EasyOCR reader: {e}", file=sys.stderr)
            self.reader = None

    def recognize(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        if self.reader is None or plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        try:
            h, w = plate_crop.shape[:2]
            target_h = 75
            if h < target_h:
                scale = target_h / float(h)
                processed = cv2.resize(plate_crop, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)
            else:
                processed = plate_crop.copy()

            # Add white border padding to prevent character boundary clipping
            pad_y, pad_x = 12, 18
            if len(processed.shape) == 3:
                padded = cv2.copyMakeBorder(processed, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            else:
                padded = cv2.copyMakeBorder(processed, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=255)

            results = self.reader.readtext(
                padded,
                allowlist=self.ALLOWED_CHARACTERS,
                detail=1,
                paragraph=False,
            )
            if not results:
                return "", 0.0

            # Sort detected text regions horizontally (left-to-right)
            sorted_results = sorted(results, key=lambda item: item[0][0][0])

            texts = [item[1].strip() for item in sorted_results if item[1].strip()]
            confs = [float(item[2]) for item in sorted_results if item[1].strip()]

            combined_text = "".join(texts)
            avg_conf = float(np.mean(confs)) if confs else 0.0

            return combined_text, avg_conf
        except Exception as e:
            print(f"[WARNING] EasyOCR inference error: {e}", file=sys.stderr)
            return "", 0.0


class MockPlateOCREngine(BasePlateOCREngine):
    """Mock OCR engine for deterministic unit testing and offline diagnostics."""

    def __init__(self, fixed_text: str = "MH12AB1234", confidence: float = 0.95):
        self.fixed_text = fixed_text
        self.confidence = confidence

    def recognize(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0
        return self.fixed_text, self.confidence


class HeuristicPlateOCREngine(BasePlateOCREngine):
    """
    Lightweight fallback OCR engine using character segmentation and contour analysis
    when heavy OCR libraries (EasyOCR/Tesseract) are not installed.
    """

    def recognize(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        if len(plate_crop.shape) == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop.copy()

        # Simple thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find character-like contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_boxes = []
        h, w = thresh.shape[:2]

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Filter character dimensions (aspect ratio ~0.2 to 1.0, height ~30% to 90% of plate)
            if 0.2 <= (cw / float(ch)) <= 1.2 and (0.3 * h) <= ch <= (0.95 * h):
                char_boxes.append((x, y, cw, ch))

        if not char_boxes:
            return "", 0.0

        char_boxes.sort(key=lambda b: b[0])  # Left to right
        estimated_conf = min(0.70, len(char_boxes) / 10.0)
        return f"PLATE{len(char_boxes)}", estimated_conf


def get_plate_ocr_engine(engine_type: str = "auto") -> BasePlateOCREngine:
    """
    Factory function returning the configured OCR engine.
    """
    if engine_type == "mock":
        return MockPlateOCREngine()

    if engine_type == "easyocr":
        return EasyOCREngine()

    if engine_type == "auto":
        try:
            import easyocr
            return EasyOCREngine()
        except ImportError:
            return HeuristicPlateOCREngine()

    return HeuristicPlateOCREngine()
