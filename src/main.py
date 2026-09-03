import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path when running main.py directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from src.core.types import FaceDetection, IdentityMatch, PlateRecognitionResult, Track
from src.anpr import format_indian_plate
from src.events import Zone
from src.face.association import FaceTrackAssociation
from src.pipeline import PipelineSession, PipelineSubsystemError

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
FACE_KNOWN_COLOR = (0, 255, 127)    # Spring Green for a recognized identity match
FACE_UNKNOWN_COLOR = (255, 0, 255)  # Magenta for an unassociated face (no track yet)
FACE_NOMATCH_COLOR = (0, 0, 255)    # Red for a face that was evaluated and did NOT match the gallery
FACE_PENDING_COLOR = (0, 215, 255)  # Amber for a face associated with a track whose recognition hasn't completed yet
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
                # Only color the box once recognition has actually been evaluated
                # for this track — a track with no result yet keeps the default
                # person color rather than being colored as if it already failed.
                color = FACE_KNOWN_COLOR if match_info.is_match else FACE_NOMATCH_COLOR
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

        # Three genuinely distinct states — never collapsed into one "FLAGGED"
        # label: a track whose recognition hasn't run yet is not the same as
        # one that ran and didn't match the gallery.
        if match_info is not None and match_info.is_match:
            box_color = FACE_KNOWN_COLOR
            face_label = f"face -> #{assoc_track_id} | KNOWN: {match_info.identity} ({match_info.similarity:.2f})"
        elif match_info is not None:
            box_color = FACE_NOMATCH_COLOR
            face_label = f"face -> #{assoc_track_id} | NOT RECOGNIZED ({match_info.similarity:.2f})"
        elif assoc_track_id is not None:
            box_color = FACE_PENDING_COLOR
            face_label = f"face -> #{assoc_track_id} | RECOGNIZING..."
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
        default="data/videos/shreyas1.mp4",
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
        default=os.environ.get("VISION_DB_URI", os.environ.get("DATABASE_URL")),
        help="Optional PostgreSQL connection URI for persistent gallery storage (default: read from VISION_DB_URI/DATABASE_URL env vars, or disabled/in-memory if unset)",
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
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device for YOLO detection and face recognition (default: auto — CUDA if available, else CPU). "
        "YuNet face detection and ByteTrack/EventEngine always run on CPU regardless of this flag.",
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
    print(f" Device              : {args.device}")
    print("==================================================")

    # Build the shared pipeline session (video ingestion, detection, tracking,
    # face recognition, ANPR, zones/event engine). This is the single
    # implementation of the AI orchestration — the FastAPI backend drives the
    # exact same PipelineSession class, so there is no parallel pipeline here.
    try:
        session = PipelineSession(
            video_path=args.video,
            model=args.model,
            confidence=args.confidence,
            face_confidence=args.face_confidence,
            face_threshold=args.face_threshold,
            face_margin=args.face_margin,
            gallery_dir=args.gallery_dir,
            db_uri=args.db_uri,
            interval=args.interval,
            face_model=args.face_model,
            arcface_backend=args.arcface_backend,
            enable_anpr=not args.disable_anpr,
            plate_model=args.plate_model,
            ocr_engine=args.ocr_engine,
            zones_path=args.zones,
            loitering_duration=args.loitering_duration,
            stationary_duration=args.stationary_duration,
            movement_threshold=args.movement_threshold,
            debug_face_matching=args.debug_face_matching,
            debug_face_alignment=args.debug_face_alignment,
            debug_face_crops=args.debug_face_crops,
            device=args.device,
            verbose=True,
        )
    except PipelineSubsystemError as e:
        print(f"[ERROR] {e.subsystem} initialization failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error initializing pipeline: {e}", file=sys.stderr)
        sys.exit(1)

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

    frame_index = 0

    try:
        while True:
            frame_start = time.time()
            frame = session.source.read_frame()

            if frame is None:
                print("[INFO] Reached end of video stream.")
                break

            frame_index = session.source.current_frame
            session.process_frame(frame, frame_index)

            elapsed_total = time.time() - session.start_time
            actual_source_fps = (
                frame_index / elapsed_total if elapsed_total > 0 else session.source.fps
            )

            latest_alert_title: Optional[str] = None
            if session.event_engine is not None and session.event_engine.active_alerts:
                latest_alert_title = session.event_engine.active_alerts[-1].title

            # Annotate frame
            annotated_frame = draw_annotations(
                frame=frame,
                tracks=session.latest_tracks,
                faces=session.latest_faces,
                associations=session.latest_associations,
                track_identity_map=session.track_identity_cache,
                plates=session.latest_plates,
                track_plate_map=session.track_plate_map,
                zones=session.zones,
                breached_zone_ids=session.latest_breached_zone_ids,
            )

            # Calculate unique flagged tracks
            unique_flagged = sum(1 for m in session.track_identity_cache.values() if not m.is_match)
            active_vehicles_count = len([t for t in session.latest_tracks if t.class_name in TARGET_VEHICLE_CLASSES])
            active_events_count = len(session.event_engine.event_history) if session.event_engine else 0

            # Draw HUD
            final_frame = draw_hud(
                frame=annotated_frame,
                current_frame=frame_index,
                total_frames=session.source.frame_count,
                source_fps=actual_source_fps,
                inference_fps=session.recent_inference_fps,
                active_tracks_count=len(session.latest_tracks),
                detection_count=len(session.latest_detections),
                faces_detected_count=len(session.latest_faces),
                faces_associated_count=len(session.latest_associations),
                embeddings_generated_count=session.total_embeddings_generated,
                recognized_faces_count=session.total_recognized_faces,
                flagged_tracks_count=unique_flagged,
                recog_threshold=args.face_threshold,
                recog_margin=args.face_margin,
                face_model_name=model_display_name,
                vehicles_tracked_count=active_vehicles_count,
                plates_read_count=len(session.track_plate_map),
                active_events_count=active_events_count,
                latest_alert_title=latest_alert_title,
            )

            # Display frame if GUI available
            if has_gui:
                cv2.imshow(window_name, final_frame)

                target_delay_ms = max(1, int(1000 / session.source.fps)) if session.source.fps > 0 else 30
                processing_time_ms = int((time.time() - frame_start) * 1000)
                wait_delay = max(1, target_delay_ms - processing_time_ms)

                key = cv2.waitKey(wait_delay) & 0xFF
                if key == ord("q"):
                    print("[INFO] Exit key 'q' pressed by user. Shutting down...")
                    break
            else:
                if frame_index % 25 == 0 or frame_index == session.source.frame_count:
                    total_str = f"/{session.source.frame_count}" if session.source.frame_count > 0 else ""
                    print(
                        f"[PROGRESS] Frame {frame_index}{total_str} | Tracks: {len(session.latest_tracks)} | "
                        f"Faces: {session.total_recognized_faces} | Plates: {len(session.track_plate_map)} | Inf FPS: {session.recent_inference_fps:.1f}",
                        file=sys.stderr,
                    )

    finally:
        session.release()
        if has_gui:
            cv2.destroyAllWindows()

    avg_inf_fps = (
        (session.inference_count / session.total_inference_time)
        if session.total_inference_time > 0
        else 0.0
    )
    print("==================================================")
    print("VISION — Slice 7.0 Summary (Events + Face + ANPR)")
    print("==================================================")
    print(f"Frames Processed       : {frame_index}")
    print(f"YOLO Inferences        : {session.inference_count}")
    print(f"Total Detections       : {session.total_detections}")
    print(f"Unique Tracks          : {len(session.observed_unique_track_ids)}")
    print(f"Max Active Tracks      : {session.max_active_tracks}")
    print(f"Faces Detected         : {session.total_faces_detected}")
    print(f"Faces Associated       : {session.total_faces_associated}")
    print(f"Embeddings Generated   : {session.total_embeddings_generated}")
    print(f"Recognized Faces       : {session.total_recognized_faces}")
    print(f"Unknown Faces          : {session.total_unknown_faces}")
    if session.enable_anpr:
        print(f"ANPR Engine            : Enabled ({args.ocr_engine})")
        print(f"Plates Detected        : {session.total_plates_detected}")
        print(f"Unique Plates Read     : {len(session.track_plate_map)}")
        for tid, prec in session.track_plate_map.items():
            print(f"  • Vehicle Track #{tid}: {format_indian_plate(prec.cleaned_text)} (conf: {prec.confidence:.2f}, valid: {prec.is_valid})")
    if session.event_engine is not None:
        print(f"Surveillance Zones     : {len(session.zones)}")
        print(f"Total Security Events  : {len(session.event_engine.event_history)}")
        print(f"Total Alerts Emitted   : {len(session.event_engine.active_alerts)}")
        event_counts = {}
        for ev in session.event_engine.event_history:
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

