"""
Automatic Number Plate Recognition (ANPR) & Vehicle Intelligence Package.
Provides license plate detection, contrast enhancement, OCR, Indian registration
syntax disambiguation, vehicle-track association, and temporal confidence pooling.
"""

from src.anpr.association import associate_plates_to_vehicles, map_crop_to_global_bbox
from src.anpr.cache import PlateTrackCache, TrackPlateRecord
from src.anpr.cleaner import (
    clean_plate_text,
    disambiguate_indian_plate,
    format_indian_plate,
    is_valid_generic_plate,
    is_valid_indian_plate,
    strip_non_alphanumeric,
)
from src.anpr.detector import LicensePlateDetector
from src.anpr.enhancer import PlateEnhancer, enhance_plate_crop
from src.anpr.ocr import (
    BasePlateOCREngine,
    EasyOCREngine,
    HeuristicPlateOCREngine,
    MockPlateOCREngine,
    get_plate_ocr_engine,
)

__all__ = [
    "LicensePlateDetector",
    "PlateEnhancer",
    "enhance_plate_crop",
    "BasePlateOCREngine",
    "EasyOCREngine",
    "HeuristicPlateOCREngine",
    "MockPlateOCREngine",
    "get_plate_ocr_engine",
    "clean_plate_text",
    "disambiguate_indian_plate",
    "format_indian_plate",
    "is_valid_indian_plate",
    "is_valid_generic_plate",
    "strip_non_alphanumeric",
    "associate_plates_to_vehicles",
    "map_crop_to_global_bbox",
    "PlateTrackCache",
    "TrackPlateRecord",
]
