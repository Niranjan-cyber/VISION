from typing import Optional, Tuple
import cv2
import numpy as np


class PlateEnhancer:
    """
    Enhances license plate crops for robust character segmentation and OCR.
    Applies CLAHE contrast balancing, bilateral edge-preserving smoothing,
    and optional deskewing.
    """

    def __init__(
        self,
        target_height: int = 70,
        enable_clahe: bool = True,
        enable_bilateral: bool = True,
        enable_deskew: bool = True,
    ):
        self.target_height = target_height
        self.enable_clahe = enable_clahe
        self.enable_bilateral = enable_bilateral
        self.enable_deskew = enable_deskew

        if self.enable_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def enhance(self, plate_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Enhances an input BGR or Grayscale license plate crop.

        Args:
            plate_crop: np.ndarray image crop of the license plate.

        Returns:
            Enhanced BGR/Grayscale image ready for OCR, or None if crop is invalid.
        """
        if plate_crop is None or not isinstance(plate_crop, np.ndarray) or plate_crop.size == 0:
            return None

        h, w = plate_crop.shape[:2]
        if h < 8 or w < 16:
            return None

        # 1. Convert to grayscale if needed
        if len(plate_crop.shape) == 3 and plate_crop.shape[2] == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        elif len(plate_crop.shape) == 2:
            gray = plate_crop.copy()
        else:
            return None

        # 2. Resize to standard height maintaining aspect ratio
        if self.target_height > 0 and h != self.target_height:
            scale = self.target_height / float(h)
            new_w = max(16, int(round(w * scale)))
            gray = cv2.resize(gray, (new_w, self.target_height), interpolation=cv2.INTER_CUBIC)

        # 3. Deskewing if enabled
        if self.enable_deskew:
            gray = self._deskew(gray)

        # 4. Bilateral filtering for edge-preserving denoising
        if self.enable_bilateral:
            gray = cv2.bilateralFilter(gray, d=7, sigmaColor=60, sigmaSpace=60)

        # 5. Contrast Limited Adaptive Histogram Equalization
        if self.enable_clahe and self.clahe is not None:
            gray = self.clahe.apply(gray)

        return gray

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """Corrects minor angular tilt (-25 to +25 degrees) based on horizontal edges."""
        h, w = gray.shape[:2]
        if h < 15 or w < 30:
            return gray

        # Otsu threshold to detect plate characters / border orientation
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find coordinates of all foreground pixels
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50:
            return gray

        # Compute minimum area bounding box
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # Normalize OpenCV minAreaRect angle convention
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Only deskew if tilt is reasonable (between -25 and +25 degrees)
        if abs(angle) < 1.0 or abs(angle) > 25.0:
            return gray

        # Rotate around center
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(
            gray,
            rot_mat,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return deskewed


def enhance_plate_crop(plate_crop: np.ndarray) -> Optional[np.ndarray]:
    """Convenience function running default PlateEnhancer pipeline."""
    enhancer = PlateEnhancer()
    return enhancer.enhance(plate_crop)
