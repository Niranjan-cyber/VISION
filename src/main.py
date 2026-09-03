import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path when running main.py directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from src.core.types import (
    BoundingBox,
    Detection,
    FaceDetection,
    FaceEmbedding,
    IdentityMatch,
    PlateDetection,
    PlateRecognitionResult,
    Track,
    VehiclePlateAssociation,
)
from src.anpr import (
    LicensePlateDetector,
    PlateEnhancer,
    PlateTrackCache,
    clean_plate_text,
    format_indian_plate,
    get_plate_ocr_engine,
    map_crop_to_global_bbox,
)
from src.detection.detector import YOLODetector
from src.events import (
    Alert,
    EventEngine,
    EventType,
    ObjectState,
    SecurityEvent,
    Severity,
    Zone,
    load_zones_from_file,
    point_in_zone,
)
from src.face.alignment import align_face
from src.face.association import FaceTrackAssociation, associate_faces_to_tracks
from src.face.detector import FaceDetector
from src.face.embedder import (
    FaceEmbedder,
    ONNXRuntimeArcFaceEmbedder,
    OpenCVArcFaceEmbedder,
)
from src.face.gallery import load_gallery_from_dir
from src.face.matcher import FaceMatcher
from src.face.modern_embedder import W600KR50Embedder
from src.ingestion.video import VideoSource
from src.tracking.tracker import ByteTrackTracker

# Target classes for vehicle intelligence
TARGET_VEHICLE_CLASSES: Set[str] = {"car", "truck", "bus", "motorcycle"}
PLATE_BOX_COLOR = (0, 215, 255)  # Amber/Gold for license plates

# Color palette for object visualization (BGR format)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "person": (0, 215, 255),     # Bright Amber/Gold
    "car": (0, 255, 127),        # Spring Green
    "truck": (255, 144, 30),     # Dodger Blue
    "bus": (211, 0, 148),        # Deep Purple
    "motorcycle": (255, 255, 0), # Cyan
    "bicycle": (0, 255, 255),    # Bright Yellow
}
FACE_KNOWN_COLOR = (0, 255, 127)   # Spring Green for recognized identity match
FACE_UNKNOWN_COLOR = (255, 0, 255) # Magenta for unknown / unassociated face
FACE_FLAGGED_COLOR = (0, 0, 255)   # Red for flagged / unrecognized intruder
DEFAULT_COLOR = (200, 200, 200)


def draw_annotations(
    frame: np.ndarray,
    tracks: List[Track],
    faces: List[FaceDetection],
    associations: List[FaceTrackAssociation],
    track_identity_map: Dict[int, IdentityMatch],
    plates: Optional[List[PlateRecognitionResult]] = None,
    track_plate_map: Optional[Dict[int, PlateRecognitionResult]] = None,
    zones: Optional[List[Zone]] = None,
    breached_zone_ids: Optional[Set[str]] = None,
) -> np.ndarray:
    """Renders surveillance zones, tracked bounding boxes, face boxes, landmarks, vehicle plates, and status indicators."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 0. Render Surveillance Zones (Polygons, Semi-transparent Fill, and Labels)
    if zones:
        for zone in zones:
            if len(zone.polygon) < 3:
                continue
            is_breached = (breached_zone_ids is not None and zone.id in breached_zone_ids)
            zone_color = (0, 0, 255) if is_breached else ((0, 200, 0) if zone.zone_type == "restricted" else (0, 215, 255))

            # Semi-transparent overlay fill
            pts = np.array(zone.polygon, dtype=np.int32)
            poly_overlay = annotated.copy()
            cv2.fillPoly(poly_overlay, [pts], zone_color)
            alpha = 0.35 if is_breached else 0.18
            cv2.addWeighted(poly_overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

            # Border line
            cv2.polylines(annotated, [pts], isClosed=True, color=zone_color, thickness=2)

            # Label at first vertex
            lx, ly = zone.polygon[0]
            status_tag = " [BREACHED]" if is_breached else f" [{zone.zone_type.upper()}]"
            zone_lbl = f"{zone.name}{status_tag}"
            cv2.putText(
                annotated,
                zone_lbl,
                (lx + 6, max(18, ly - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                zone_color,
                1,
                lineType=cv2.LINE_AA,
            )

    # Map face object ID -> associated track_id
    associated_map: Dict[int, int] = {
        id(assoc.face): assoc.track_id for assoc in associations
    }

    # 1. Render Tracked Objects (Vehicles & Persons)
    for trk in tracks:
        color = CLASS_COLORS.get(trk.class_name, DEFAULT_COLOR)
        
        # Visually highlight person track status or vehicle license plate:
        if trk.class_name == "person":
            match_info = track_identity_map.get(trk.track_id)
            if match_info is not None:
                if match_info.is_match:
                    color = FACE_KNOWN_COLOR  # Green for valid/authorized
                else:
                    color = FACE_FLAGGED_COLOR  # Red for FLAGGED/unauthorized
            label = f"{trk.class_name} #{trk.track_id} {trk.confidence:.2f}"
        elif trk.class_name in TARGET_VEHICLE_CLASSES and track_plate_map and trk.track_id in track_plate_map:
            p_rec = track_plate_map[trk.track_id]
            label = f"{trk.class_name} #{trk.track_id} | {format_indian_plate(p_rec.cleaned_text)}"
        else:
            label = f"{trk.class_name} #{trk.track_id} {trk.confidence:.2f}"

        bbox = trk.bbox

        x1 = max(0, min(bbox.x1, w - 1))
        y1 = max(0, min(bbox.y1, h - 1))
        x2 = max(0, min(bbox.x2, w - 1))
        y2 = max(0, min(bbox.y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_y1 = max(y1 - text_h - 6, 0)
        label_y2 = label_y1 + text_h + 6
        label_x2 = min(x1 + text_w + 8, w)

        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, cv2.FILLED)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y2 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

    # 2. Render Face Boxes, Landmarks & Identity Match Labels
    for face in faces:
        bbox = face.bbox
        fx1 = max(0, min(bbox.x1, w - 1))
        fy1 = max(0, min(bbox.y1, h - 1))
        fx2 = max(0, min(bbox.x2, w - 1))
        fy2 = max(0, min(bbox.y2, h - 1))

        if fx2 <= fx1 or fy2 <= fy1:
            continue

        assoc_track_id = associated_map.get(id(face))
        match_info = track_identity_map.get(assoc_track_id) if assoc_track_id is not None else None

        if match_info is not None and match_info.is_match:
            box_color = FACE_KNOWN_COLOR
            face_label = f"face -> #{assoc_track_id} | {match_info.identity} ({match_info.similarity:.2f})"
        elif assoc_track_id is not None:
            box_color = FACE_FLAGGED_COLOR
            sim_str = f" ({match_info.similarity:.2f})" if match_info is not None else ""
            face_label = f"face -> #{assoc_track_id} | FLAGGED{sim_str}"
        else:
            box_color = FACE_UNKNOWN_COLOR
            face_label = f"face {face.confidence:.2f}"

        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), box_color, thickness=2)

        # Draw 5 facial landmarks if available
        if face.landmarks is not None:
            for pt in face.landmarks:
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(face_label, font, font_scale, thickness)

        label_y1 = max(fy1 - text_h - 4, 0)
        label_y2 = label_y1 + text_h + 4
        label_x2 = min(fx1 + text_w + 6, w)

        cv2.rectangle(annotated, (fx1, label_y1), (label_x2, label_y2), box_color, cv2.FILLED)
        cv2.putText(
            annotated,
            face_label,
            (fx1 + 3, label_y2 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0) if (match_info and match_info.is_match) else (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )

    # 3. Render Detected License Plates
    if plates:
        for plt in plates:
            pb = plt.bbox
            px1 = max(0, min(pb.x1, w - 1))
            py1 = max(0, min(pb.y1, h - 1))
            px2 = max(0, min(pb.x2, w - 1))
            py2 = max(0, min(pb.y2, h - 1))
            if px2 <= px1 or py2 <= py1:
                continue

            cv2.rectangle(annotated, (px1, py1), (px2, py2), PLATE_BOX_COLOR, thickness=2)
            plt_label = f"PLATE: {format_indian_plate(plt.cleaned_text)} ({plt.confidence:.2f})"
            (tw, th), bl = cv2.getTextSize(plt_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            ly1 = max(py1 - th - 4, 0)
            ly2 = ly1 + th + 4
            lx2 = min(px1 + tw + 6, w)
            cv2.rectangle(annotated, (px1, ly1), (lx2, ly2), PLATE_BOX_COLOR, cv2.FILLED)
            cv2.putText(
                annotated,
                plt_label,
                (px1 + 3, ly2 - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA,
            )

    return annotated


def draw_hud(
    frame: np.ndarray,
    current_frame: int,
    total_frames: int,
    source_fps: float,
    inference_fps: float,
    active_tracks_count: int,
    detection_count: int,
    faces_detected_count: int,
    faces_associated_count: int,
    embeddings_generated_count: int,
    recognized_faces_count: int,
    flagged_tracks_count: int,
    recog_threshold: float,
    recog_margin: float,
    face_model_name: str,
    vehicles_tracked_count: int = 0,
    plates_read_count: int = 0,
    active_events_count: int = 0,
    latest_alert_title: Optional[str] = None,
) -> np.ndarray:
    """Draws runtime status HUD on top-left of frame."""
    hud_frame = frame.copy()
    overlay = hud_frame.copy()

    panel_x1, panel_y1 = 15, 15
    panel_x2, panel_y2 = 320, 440

    # Glassmorphic dark panel background
    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (20, 20, 20),
        cv2.FILLED,
    )
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, hud_frame, 1 - alpha, 0, hud_frame)
    border_color = (0, 0, 255) if active_events_count > 0 else (0, 215, 255)
    cv2.rectangle(
        hud_frame,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        border_color,
        2 if active_events_count > 0 else 1,
    )

    lines = [
        ("VISION - Slice 7.0 (Events & Alerts)", (0, 215, 255), 0.48, 2),
        (
            f"Frame: {current_frame} / {total_frames if total_frames > 0 else 'N/A'}",
            (220, 220, 220),
            0.42,
            1,
        ),
        (f"Source FPS: {source_fps:.1f}", (220, 220, 220), 0.42, 1),
        (
            f"Inference FPS: {inference_fps:.1f}",
            (0, 255, 127) if inference_fps > 0 else (150, 150, 150),
            0.42,
            1,
        ),
        (f"Active Tracks: {active_tracks_count}", (0, 215, 255), 0.42, 1),
        (f"Detections: {detection_count}", (220, 220, 220), 0.42, 1),
        (f"Faces Detected: {faces_detected_count}", (255, 0, 255), 0.42, 1),
        (f"Faces Associated: {faces_associated_count}", (0, 255, 255), 0.42, 1),
        (f"Embeddings Gen: {embeddings_generated_count}", (200, 200, 200), 0.42, 1),
        (f"Recognized Faces: {recognized_faces_count}", (0, 255, 127), 0.42, 1),
        (f"Flagged Tracks: {flagged_tracks_count}", (0, 0, 255), 0.42, 1),
        (f"Vehicles: {vehicles_tracked_count}", (255, 144, 30), 0.42, 1),
        (f"Plates Read: {plates_read_count}", (0, 215, 255), 0.42, 1),
        (
            f"Active Events: {active_events_count}",
            (0, 0, 255) if active_events_count > 0 else (0, 255, 127),
            0.44,
            2 if active_events_count > 0 else 1,
        ),
        (f"Face Model: {face_model_name}", (0, 215, 255), 0.40, 1),
        ("Alignment: 5-Point Similarity", (0, 255, 255), 0.40, 1),
        ("Embedding: 512-D L2 Norm", (200, 200, 200), 0.40, 1),
        (f"Threshold: {recog_threshold:.2f} | Margin: {recog_margin:.2f}", (200, 200, 200), 0.40, 1),
    ]

    if latest_alert_title:
        lines.insert(1, (f"ALERT: {latest_alert_title[:24]}", (0, 0, 255), 0.42, 2))

    y_offset = panel_y1 + 18
    for text, color, scale, thickness in lines:
        cv2.putText(
            hud_frame,
            text,
            (panel_x1 + 10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        y_offset += 20

    return hud_frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="VISION Vertical Slice 7.0 - Event Intelligence & Alert Engine Pipeline"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/videos/test.mp4",
        help="Path to input video file (MP4 format)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO object detection (default: 0.25)",
    )
    parser.add_argument(
        "--face-confidence",
        type=float,
        default=0.50,
        help="Confidence threshold for Face detection (default: 0.50)",
    )
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=0.60,
        help="Cosine similarity threshold for Face Recognition (default: 0.60)",
    )
    parser.add_argument(
        "--face-margin",
        type=float,
        default=0.10,
        help="Minimum similarity margin between best and second-best candidate (default: 0.10)",
    )
    parser.add_argument(
        "--gallery-dir",
        type=str,
        default="data/face_gallery",
        help="Path to face gallery directory (default: data/face_gallery)",
    )
    parser.add_argument(
        "--db-uri",
        type=str,
        default=os.environ.get("VISION_DB_URI", os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vision_db")),
        help="PostgreSQL connection URI (default: read from VISION_DB_URI/DATABASE_URL or postgresql://postgres:postgres@localhost:5432/vision_db)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Inference frame interval N (run YOLO every Nth frame, default: 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLO model path/name (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--face-model",
        type=str,
        choices=["w600k_r50", "r100"],
        default="w600k_r50",
        help="Recognition model architecture (default: w600k_r50)",
    )
    parser.add_argument(
        "--arcface-backend",
        type=str,
        choices=["onnxruntime", "opencv"],
        default="onnxruntime",
        help="Inference backend for legacy ArcFace R100 model (default: onnxruntime)",
    )
    parser.add_argument(
        "--debug-face-matching",
        action="store_true",
        help="Print diagnostic log of candidate similarity scores for newly evaluated track embeddings",
    )
    parser.add_argument(
        "--debug-face-alignment",
        action="store_true",
        help="Save aligned face crop images to data/debug/aligned_faces/ for visual verification",
    )
    parser.add_argument(
        "--debug-face-crops",
        action="store_true",
        help="Save raw extracted face crop images to scratch/debug_face_crops/ for visual inspection",
    )
    parser.add_argument(
        "--disable-anpr",
        action="store_true",
        help="Disable automatic number plate recognition on vehicle tracks",
    )
    parser.add_argument(
        "--plate-model",
        type=str,
        default=None,
        help="Path to custom YOLO license plate detection model weights (.pt or .onnx)",
    )
    parser.add_argument(
        "--ocr-engine",
        type=str,
        choices=["auto", "easyocr", "heuristic", "mock"],
        default="auto",
        help="OCR engine for license plate text extraction (default: auto)",
    )
    parser.add_argument(
        "--zones",
        type=str,
        default=None,
        help="Path to YAML surveillance zone configuration file (e.g. configs/zones.yaml)",
    )
    parser.add_argument(
        "--loitering-duration",
        type=float,
        default=30.0,
        help="Dwell duration in seconds before triggering LOITERING for persons in restricted zones (default: 30.0)",
    )
    parser.add_argument(
        "--stationary-duration",
        type=float,
        default=60.0,
        help="Dwell duration in seconds before triggering SUSPICIOUS_VEHICLE for stopped vehicles (default: 60.0)",
    )
    parser.add_argument(
        "--movement-threshold",
        type=float,
        default=15.0,
        help="Displacement threshold in pixels below which an object is considered stationary (default: 15.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_display_name = "InsightFace W600K-R50" if args.face_model == "w600k_r50" else f"ArcFace R100 ({args.arcface_backend})"

    print("==================================================")
    print("       VISION — Vertical Slice 7.0 Pipeline       ")
    print("==================================================")
    print(f" Video Path          : {args.video}")
    print(f" YOLO Model          : {args.model}")
    print(f" ANPR Enabled        : {not args.disable_anpr}")
    if not args.disable_anpr:
        print(f" OCR Engine          : {args.ocr_engine}")
    if args.zones:
        print(f" Zones Config        : {args.zones}")
        print(f" Loitering Duration  : {args.loitering_duration:.1f}s")
        print(f" Stationary Duration : {args.stationary_duration:.1f}s")
    print(f" Recognition Model   : {model_display_name}")
    print(f" Embedding Dim       : 512-D L2-Normalized")
    print(f" Alignment           : 5-Point Landmark Similarity Transform")
    print(f" Preprocessing       : 112x112 BGR / 127.5 Normalization")
    print(f" Confidence Threshold: {args.confidence}")
    print(f" Face Confidence     : {args.face_confidence}")
    print(f" Recognition Thresh  : {args.face_threshold:.2f}")
    print(f" Recognition Margin  : {args.face_margin:.2f}")
    print(f" Gallery Directory   : {args.gallery_dir}")
    print(f" Debug Matching      : {args.debug_face_matching}")
    print(f" Debug Alignment     : {args.debug_face_alignment}")
    print(f" Inference Interval  : Every {args.interval} frame(s)")
    print("==================================================")

    if args.debug_face_crops:
        os.makedirs("scratch/debug_face_crops", exist_ok=True)
    if args.debug_face_alignment:
        os.makedirs("data/debug/aligned_faces", exist_ok=True)

    # 1. Ingestion Initialization
    try:
        source = VideoSource(args.video)
    except FileNotFoundError as fnf_err:
        print(f"[ERROR] {fnf_err}", file=sys.stderr)
        sys.exit(1)
    except ValueError as val_err:
        print(f"[ERROR] {val_err}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error opening video: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[INFO] Video loaded successfully. Resolution: {source.width}x{source.height}, "
        f"FPS: {source.fps:.2f}, Total Frames: {source.frame_count}"
    )

    # 2. YOLODetector Initialization
    try:
        detector = YOLODetector(
            model_name=args.model,
            confidence_threshold=args.confidence,
        )
    except Exception as e:
        print(f"[ERROR] Detector initialization failed: {e}", file=sys.stderr)
        source.release()
        sys.exit(1)

    # 3. ByteTrackTracker Initialization
    try:
        tracker = ByteTrackTracker(
            track_thresh=args.confidence,
            match_thresh=0.8,
            track_buffer=30,
        )
    except Exception as e:
        print(f"[ERROR] Tracker initialization failed: {e}", file=sys.stderr)
        source.release()
        sys.exit(1)

    # 4. FaceDetector Initialization
    try:
        face_detector = FaceDetector(score_threshold=args.face_confidence)
    except Exception as e:
        print(f"[ERROR] FaceDetector initialization failed: {e}", file=sys.stderr)
        source.release()
        sys.exit(1)

    # 5. FaceEmbedder Initialization based on --face-model
    try:
        if args.face_model == "w600k_r50":
            face_embedder = W600KR50Embedder(model_path="models/w600k_r50.onnx")
        else:
            if args.arcface_backend == "opencv":
                face_embedder = OpenCVArcFaceEmbedder()
            else:
                face_embedder = ONNXRuntimeArcFaceEmbedder()
    except Exception as e:
        print(f"[ERROR] FaceEmbedder initialization failed: {e}", file=sys.stderr)
        source.release()
        sys.exit(1)

    # 6. FaceGallery & FaceMatcher Initialization
    try:
        print(f"[INFO] Connecting to database: {args.db_uri}", file=sys.stderr)
        gallery = load_gallery_from_dir(
            args.gallery_dir, face_detector, face_embedder, db_uri=args.db_uri
        )
        face_matcher = FaceMatcher(
            gallery, threshold=args.face_threshold, margin=args.face_margin
        )
    except Exception as e:
        print(f"[WARNING] PostgreSQL database initialization failed ({e}). Falling back to in-memory mode.", file=sys.stderr)
        try:
            gallery = load_gallery_from_dir(args.gallery_dir, face_detector, face_embedder, db_uri=None)
            face_matcher = FaceMatcher(gallery, threshold=args.face_threshold, margin=args.face_margin)
        except Exception as fallback_err:
            print(f"[ERROR] Face Gallery/Matcher fallback initialization failed: {fallback_err}", file=sys.stderr)
            source.release()
            sys.exit(1)

    # 7. ANPR Engine Initialization
    enable_anpr = not args.disable_anpr
    if enable_anpr:
        plate_detector = LicensePlateDetector(model_path=args.plate_model)
        plate_enhancer = PlateEnhancer(target_height=70)
        plate_ocr = get_plate_ocr_engine(args.ocr_engine)
        plate_track_cache = PlateTrackCache()
    else:
        plate_detector = None
        plate_enhancer = None
        plate_ocr = None
        plate_track_cache = None

    # 8. Event Intelligence Engine & Zones Initialization
    event_engine = None
    configured_zones = []
    if args.zones:
        try:
            configured_zones = load_zones_from_file(args.zones)
            print(f"[INFO] Loaded {len(configured_zones)} surveillance zone(s) from '{args.zones}'.")
            event_engine = EventEngine(
                zones=configured_zones,
                loitering_duration=args.loitering_duration,
                stationary_duration=args.stationary_duration,
                movement_threshold=args.movement_threshold,
            )
        except Exception as e:
            print(f"[WARNING] Failed to load zones from '{args.zones}': {e}. Event engine disabled.", file=sys.stderr)

    window_name = "VISION - Vertical Slice 7.0 (Events & Alerts)"
    has_gui = True
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error:
        print(
            "[WARNING] OpenCV GUI windowing is unavailable (opencv-python-headless detected). "
            "Running in headless console mode. (To enable GUI playback, activate .venv with: .\\.venv\\Scripts\\Activate.ps1)",
            file=sys.stderr,
        )
        has_gui = False

    latest_detections: List[Detection] = []
    latest_tracks: List[Track] = []
    latest_faces: List[FaceDetection] = []
    latest_associations: List[FaceTrackAssociation] = []
    latest_plates: List[PlateRecognitionResult] = []
    track_plate_map: Dict[int, PlateRecognitionResult] = {}
    total_plates_detected = 0

    # Track-level identity cache: track_id -> IdentityMatch
    track_identity_cache: Dict[int, IdentityMatch] = {}
    aligned_debug_saved_count = 0

    frame_index = 0
    inference_count = 0
    total_inference_time = 0.0
    recent_inference_fps = 0.0

    total_detections = 0
    total_faces_detected = 0
    total_faces_associated = 0
    total_embeddings_generated = 0
    total_recognized_faces = 0
    total_unknown_faces = 0
    observed_unique_track_ids: Set[int] = set()
    max_active_tracks = 0

    start_time = time.time()

    try:
        while True:
            frame_start = time.time()
            frame = source.read_frame()

            if frame is None:
                print("[INFO] Reached end of video stream.")
                break

            frame_index = source.current_frame
            frame_h, frame_w = frame.shape[:2]

            # Frame Sampling: Run YOLO, Face Detection, Embedding & Recognition every Nth frame
            if (frame_index - 1) % args.interval == 0:
                t0 = time.time()

                # A. YOLO Detection
                latest_detections = detector.detect(frame)

                # B. ByteTrack Tracking
                latest_tracks = tracker.update(latest_detections, frame_index)

                # C. Person-Only Face Detection & Coordinate Conversion
                person_tracks = [t for t in latest_tracks if t.class_name == "person"]
                current_frame_faces: List[FaceDetection] = []

                for person in person_tracks:
                    pb = person.bbox
                    px1 = max(0, min(pb.x1, frame_w))
                    py1 = max(0, min(pb.y1, frame_h))
                    px2 = max(0, min(pb.x2, frame_w))
                    py2 = max(0, min(pb.y2, frame_h))

                    if px2 <= px1 or py2 <= py1:
                        continue

                    person_crop = frame[py1:py2, px1:px2]
                    crop_faces = face_detector.detect(person_crop)

                    for crop_face in crop_faces:
                        fb = crop_face.bbox
                        gx1 = max(0, min(px1 + fb.x1, frame_w))
                        gy1 = max(0, min(py1 + fb.y1, frame_h))
                        gx2 = max(0, min(px1 + fb.x2, frame_w))
                        gy2 = max(0, min(py1 + fb.y2, frame_h))

                        if gx2 <= gx1 or gy2 <= gy1:
                            continue

                        # Convert crop-relative landmarks to global frame coordinates
                        global_landmarks = None
                        if crop_face.landmarks is not None:
                            global_landmarks = crop_face.landmarks.copy()
                            global_landmarks[:, 0] += px1
                            global_landmarks[:, 1] += py1

                        global_face = FaceDetection(
                            bbox=BoundingBox(x1=gx1, y1=gy1, x2=gx2, y2=gy2),
                            confidence=crop_face.confidence,
                            landmarks=global_landmarks,
                        )
                        current_frame_faces.append(global_face)

                latest_faces = current_frame_faces

                # D. Face-to-Track Association
                latest_associations = associate_faces_to_tracks(latest_tracks, latest_faces)

                # E. Face Recognition for Associated Faces
                frame_recognized_count = 0
                frame_unknown_count = 0

                for assoc in latest_associations:
                    track_id = assoc.track_id

                    # Check track-level identity cache
                    if track_id in track_identity_cache:
                        match_result = track_identity_cache[track_id]
                    else:
                        # 1. Attempt 5-point landmark similarity warp alignment
                        aligned_crop = None
                        if assoc.face.landmarks is not None:
                            aligned_crop = align_face(frame, assoc.face.landmarks)

                        if aligned_crop is not None:
                            face_input = aligned_crop
                        else:
                            fb = assoc.face.bbox
                            fx1 = max(0, min(fb.x1, frame_w))
                            fy1 = max(0, min(fb.y1, frame_h))
                            fx2 = max(0, min(fb.x2, frame_w))
                            fy2 = max(0, min(fb.y2, frame_h))

                            if fx2 <= fx1 or fy2 <= fy1:
                                continue

                            face_input = frame[fy1:fy2, fx1:fx2]

                        if face_input.size == 0:
                            continue

                        if args.debug_face_alignment and aligned_debug_saved_count < 5:
                            aligned_save_path = f"data/debug/aligned_faces/track_{track_id}_frame_{frame_index}.jpg"
                            cv2.imwrite(aligned_save_path, face_input)
                            aligned_debug_saved_count += 1
                            print(f"[DEBUG] Saved aligned face ({face_input.shape[1]}x{face_input.shape[0]}) to '{aligned_save_path}'")

                        if args.debug_face_crops:
                            crop_save_path = f"scratch/debug_face_crops/track_{track_id}_frame_{frame_index}.jpg"
                            cv2.imwrite(crop_save_path, face_input)

                        embedding = face_embedder.embed(face_input)
                        if embedding is not None:
                            total_embeddings_generated += 1
                            match_result = face_matcher.match(embedding)
                            track_identity_cache[track_id] = match_result

                            if args.debug_face_matching:
                                all_sims = face_matcher.get_all_similarities(embedding)
                                print(f"\n--- Diagnostic Matching for Track #{track_id} (Frame {frame_index}) ---")
                                for id_name, id_score in sorted(all_sims.items(), key=lambda x: x[1], reverse=True):
                                    print(f"  {id_name}: {id_score:.4f}")
                                print(f"  Best       : {match_result.identity if match_result.is_match else 'None'} ({match_result.similarity:.4f})")
                                print(f"  Second-Best: {match_result.second_similarity:.4f}")
                                print(f"  Margin     : {match_result.margin:.4f} (Required: >= {args.face_margin:.2f})")
                                print(f"  Match Result: {'MATCH' if match_result.is_match else 'UNKNOWN'}")

                            # Flag unrecognized/unknown users immediately
                            if not match_result.is_match:
                                print(f"[ALERT] Track #{track_id} is UNKNOWN and has been FLAGGED!", file=sys.stderr)
                                if getattr(gallery, "db", None) is not None:
                                    try:
                                        vec = embedding.vector if isinstance(embedding, FaceEmbedding) else embedding
                                        gallery.db.flag_unauthorized_user(
                                            embedding=vec,
                                            frame=face_crop,
                                            track_id=track_id,
                                            video_source=args.video
                                        )
                                        print(f"[INFO] Logged flagged event for track #{track_id} in PostgreSQL.", file=sys.stderr)
                                    except Exception as db_err:
                                        print(f"[WARNING] Failed to store flagged event in database: {db_err}", file=sys.stderr)
                        else:
                            match_result = IdentityMatch(identity=None, similarity=0.0, is_match=False)

                    if match_result.is_match:
                        frame_recognized_count += 1
                    else:
                        frame_unknown_count += 1

                total_recognized_faces += frame_recognized_count
                total_unknown_faces += frame_unknown_count

                # F. Vehicle License Plate Recognition (ANPR)
                current_frame_plates: List[PlateRecognitionResult] = []

                if enable_anpr and plate_detector is not None:
                    vehicle_tracks = [t for t in latest_tracks if t.class_name in TARGET_VEHICLE_CLASSES]
                    for veh in vehicle_tracks:
                        tid = veh.track_id
                        cached_p = plate_track_cache.get(tid)

                        # If already cached with high confidence, reuse to save compute
                        if cached_p is not None and cached_p.confidence >= 0.85:
                            current_frame_plates.append(cached_p)
                            track_plate_map[tid] = cached_p
                            continue

                        vb = veh.bbox
                        vx1 = max(0, min(vb.x1, frame_w))
                        vy1 = max(0, min(vb.y1, frame_h))
                        vx2 = max(0, min(vb.x2, frame_w))
                        vy2 = max(0, min(vb.y2, frame_h))

                        if (vx2 - vx1) < 30 or (vy2 - vy1) < 30:
                            if cached_p is not None:
                                current_frame_plates.append(cached_p)
                                track_plate_map[tid] = cached_p
                            continue

                        veh_crop = frame[vy1:vy2, vx1:vx2]
                        vh, vw = veh_crop.shape[:2]
                        plate_dets = plate_detector.detect(veh_crop)

                        best_candidate_res = None
                        for pdet in plate_dets[:3]:
                            pb = pdet.bbox
                            # Add 4px vertical and 6px horizontal margin around detected plate box
                            py1 = max(0, pb.y1 - 4)
                            py2 = min(vh, pb.y2 + 4)
                            px1 = max(0, pb.x1 - 6)
                            px2 = min(vw, pb.x2 + 6)
                            p_crop = veh_crop[py1:py2, px1:px2]
                            if p_crop.size == 0 or (px2 - px1) < 40 or (py2 - py1) < 12:
                                continue

                            enh_crop = plate_enhancer.enhance(p_crop)
                            target_crop = enh_crop if enh_crop is not None else p_crop
                            raw_txt, ocr_conf = plate_ocr.recognize(target_crop)
                            cln_txt, is_valid, mult = clean_plate_text(raw_txt)

                            if cln_txt and (is_valid or len(cln_txt) >= 7):
                                g_bbox = map_crop_to_global_bbox(
                                    BoundingBox(x1=px1, y1=py1, x2=px2, y2=py2),
                                    BoundingBox(x1=vx1, y1=vy1, x2=vx2, y2=vy2),
                                    frame_w,
                                    frame_h,
                                )
                                cand_res = PlateRecognitionResult(
                                    raw_text=raw_txt,
                                    cleaned_text=cln_txt,
                                    confidence=ocr_conf * mult,
                                    is_valid=is_valid,
                                    bbox=g_bbox,
                                )
                                # If full valid Indian plate, take it immediately
                                if is_valid and len(cln_txt) in {9, 10}:
                                    best_candidate_res = cand_res
                                    break
                                elif best_candidate_res is None or len(cln_txt) > len(best_candidate_res.cleaned_text):
                                    best_candidate_res = cand_res

                        if best_candidate_res is not None:
                            updated_res = plate_track_cache.update(tid, best_candidate_res, frame_index)
                            current_frame_plates.append(updated_res)
                            track_plate_map[tid] = updated_res
                            total_plates_detected += 1
                        elif cached_p is not None:
                            current_frame_plates.append(cached_p)
                            track_plate_map[tid] = cached_p

                    latest_plates = current_frame_plates

                t1 = time.time()

                inf_time = t1 - t0
                total_inference_time += inf_time
                inference_count += 1
                if inf_time > 0:
                    recent_inference_fps = 1.0 / inf_time

                # Metrics accumulation
                total_detections += len(latest_detections)
                total_faces_detected += len(latest_faces)
                total_faces_associated += len(latest_associations)

                for trk in latest_tracks:
                    observed_unique_track_ids.add(trk.track_id)

                if len(latest_tracks) > max_active_tracks:
                    max_active_tracks = len(latest_tracks)

            elapsed_total = time.time() - start_time
            actual_source_fps = (
                frame_index / elapsed_total if elapsed_total > 0 else source.fps
            )
            current_timestamp = (
                frame_index / source.fps if source.fps > 0 else frame_index * 0.033
            )

            # ----------------------------------------------------
            # Slice 7: Event Intelligence Evaluation
            # ----------------------------------------------------
            breached_zone_ids: Set[str] = set()
            latest_alert_title: Optional[str] = None

            if event_engine is not None:
                # Construct unified ObjectState for each active track
                object_states = []
                for trk in latest_tracks:
                    tid = trk.track_id
                    match_info = track_identity_cache.get(tid)
                    p_rec = track_plate_map.get(tid)

                    has_face = match_info is not None
                    if has_face:
                        ident = match_info.identity if match_info.is_match else "UNKNOWN"
                        sim = match_info.similarity
                    else:
                        ident = None
                        sim = None

                    plate_txt = p_rec.cleaned_text if p_rec is not None else None
                    plate_conf = p_rec.confidence if p_rec is not None else None

                    st = ObjectState(
                        track_id=tid,
                        object_type=trk.class_name,
                        bbox=trk.bbox,
                        confidence=trk.confidence,
                        camera_id="BOP-01",
                        identity=ident,
                        face_similarity=sim,
                        has_face_detected=has_face,
                        plate=plate_txt,
                        plate_confidence=plate_conf,
                        first_seen=current_timestamp,
                        last_seen=current_timestamp,
                    )
                    object_states.append(st)

                # Update deterministic event engine
                new_events, new_alerts = event_engine.update(
                    object_states, timestamp=current_timestamp
                )

                # Console notification for new alerts
                for alr in new_alerts:
                    print(
                        f"\n[SECURITY ALERT] {alr.title} | Camera: BOP-01 | "
                        f"Track: #{alr.metadata.get('object_type', 'object')} | "
                        f"Zone: {alr.metadata.get('zone_name')} | Severity: {alr.severity.value}",
                        file=sys.stderr,
                    )

                # Identify currently breached zones
                for z in configured_zones:
                    if z.zone_type == "restricted":
                        for obj in object_states:
                            if point_in_zone(obj.position, z):
                                breached_zone_ids.add(z.id)
                                break

                if event_engine.active_alerts:
                    latest_alert_title = event_engine.active_alerts[-1].title

            # Annotate frame
            annotated_frame = draw_annotations(
                frame=frame,
                tracks=latest_tracks,
                faces=latest_faces,
                associations=latest_associations,
                track_identity_map=track_identity_cache,
                plates=latest_plates,
                track_plate_map=track_plate_map,
                zones=configured_zones,
                breached_zone_ids=breached_zone_ids,
            )

            # Calculate unique flagged tracks
            unique_flagged = sum(1 for m in track_identity_cache.values() if not m.is_match)
            active_vehicles_count = len([t for t in latest_tracks if t.class_name in TARGET_VEHICLE_CLASSES])
            active_events_count = len(event_engine.event_history) if event_engine else 0

            # Draw HUD
            final_frame = draw_hud(
                frame=annotated_frame,
                current_frame=frame_index,
                total_frames=source.frame_count,
                source_fps=actual_source_fps,
                inference_fps=recent_inference_fps,
                active_tracks_count=len(latest_tracks),
                detection_count=len(latest_detections),
                faces_detected_count=len(latest_faces),
                faces_associated_count=len(latest_associations),
                embeddings_generated_count=total_embeddings_generated,
                recognized_faces_count=total_recognized_faces,
                flagged_tracks_count=unique_flagged,
                recog_threshold=args.face_threshold,
                recog_margin=args.face_margin,
                face_model_name=model_display_name,
                vehicles_tracked_count=active_vehicles_count,
                plates_read_count=len(track_plate_map),
                active_events_count=active_events_count,
                latest_alert_title=latest_alert_title,
            )

            # Display frame if GUI available
            if has_gui:
                cv2.imshow(window_name, final_frame)

                target_delay_ms = max(1, int(1000 / source.fps)) if source.fps > 0 else 30
                processing_time_ms = int((time.time() - frame_start) * 1000)
                wait_delay = max(1, target_delay_ms - processing_time_ms)

                key = cv2.waitKey(wait_delay) & 0xFF
                if key == ord("q"):
                    print("[INFO] Exit key 'q' pressed by user. Shutting down...")
                    break
            else:
                if frame_index % 25 == 0 or frame_index == source.frame_count:
                    total_str = f"/{source.frame_count}" if source.frame_count > 0 else ""
                    print(
                        f"[PROGRESS] Frame {frame_index}{total_str} | Tracks: {len(latest_tracks)} | "
                        f"Faces: {total_recognized_faces} | Plates: {len(track_plate_map)} | Inf FPS: {recent_inference_fps:.1f}",
                        file=sys.stderr,
                    )

    finally:
        source.release()
        if has_gui:
            cv2.destroyAllWindows()

    avg_inf_fps = (
        (inference_count / total_inference_time)
        if total_inference_time > 0
        else 0.0
    )
    print("==================================================")
    print("VISION — Slice 7.0 Summary (Events + Face + ANPR)")
    print("==================================================")
    print(f"Frames Processed       : {frame_index}")
    print(f"YOLO Inferences        : {inference_count}")
    print(f"Total Detections       : {total_detections}")
    print(f"Unique Tracks          : {len(observed_unique_track_ids)}")
    print(f"Max Active Tracks      : {max_active_tracks}")
    print(f"Faces Detected         : {total_faces_detected}")
    print(f"Faces Associated       : {total_faces_associated}")
    print(f"Embeddings Generated   : {total_embeddings_generated}")
    print(f"Recognized Faces       : {total_recognized_faces}")
    print(f"Unknown Faces          : {total_unknown_faces}")
    if enable_anpr:
        print(f"ANPR Engine            : Enabled ({args.ocr_engine})")
        print(f"Plates Detected        : {total_plates_detected}")
        print(f"Unique Plates Read     : {len(track_plate_map)}")
        for tid, prec in track_plate_map.items():
            print(f"  • Vehicle Track #{tid}: {format_indian_plate(prec.cleaned_text)} (conf: {prec.confidence:.2f}, valid: {prec.is_valid})")
    if event_engine is not None:
        print(f"Surveillance Zones     : {len(configured_zones)}")
        print(f"Total Security Events  : {len(event_engine.event_history)}")
        print(f"Total Alerts Emitted   : {len(event_engine.active_alerts)}")
        event_counts = {}
        for ev in event_engine.event_history:
            event_counts[ev.event_type.value] = event_counts.get(ev.event_type.value, 0) + 1
        for ev_type, count in event_counts.items():
            print(f"  • {ev_type}: {count}")
    print(f"Recognition Model      : {model_display_name}")
    print(f"Alignment              : 5-Point Landmark Similarity Transform")
    print(f"Recognition Threshold  : {args.face_threshold:.2f}")
    print(f"Recognition Margin     : {args.face_margin:.2f}")
    print(f"Average Inference FPS  : {avg_inf_fps:.2f}")
    print("==================================================")


if __name__ == "__main__":
    main()

