import argparse
import os
import sys
import time
from typing import Dict, List, Set, Tuple
import cv2
import numpy as np

from src.core.types import BoundingBox, Detection, FaceDetection, Track
from src.detection.detector import YOLODetector
from src.face.association import FaceTrackAssociation, associate_faces_to_tracks
from src.face.detector import FaceDetector
from src.ingestion.video import VideoSource
from src.tracking.tracker import ByteTrackTracker

# Color palette for object visualization (BGR format)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "person": (0, 215, 255),     # Bright Amber/Gold
    "car": (0, 255, 127),        # Spring Green
    "truck": (255, 144, 30),     # Dodger Blue
    "bus": (211, 0, 148),        # Deep Purple
    "motorcycle": (255, 255, 0), # Cyan
    "bicycle": (0, 255, 255),    # Bright Yellow
}
FACE_COLOR = (255, 0, 255)       # Magenta for faces
DEFAULT_COLOR = (200, 200, 200)


def draw_annotations(
    frame: np.ndarray,
    tracks: List[Track],
    faces: List[FaceDetection],
    associations: List[FaceTrackAssociation],
) -> np.ndarray:
    """Renders tracked object bounding boxes, face boxes, and face-to-track association labels."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Map face object ID -> associated track_id
    associated_map: Dict[int, int] = {
        id(assoc.face): assoc.track_id for assoc in associations
    }

    # 1. Render Tracked Objects (Vehicles & Persons)
    for trk in tracks:
        color = CLASS_COLORS.get(trk.class_name, DEFAULT_COLOR)
        bbox = trk.bbox

        # Safe clamp bounding box
        x1 = max(0, min(bbox.x1, w - 1))
        y1 = max(0, min(bbox.y1, h - 1))
        x2 = max(0, min(bbox.x2, w - 1))
        y2 = max(0, min(bbox.y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        # Object label: e.g. "person #17 0.94" or "car #7 0.91"
        label = f"{trk.class_name} #{trk.track_id} {trk.confidence:.2f}"
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

    # 2. Render Face Boxes & Associations
    for face in faces:
        bbox = face.bbox
        fx1 = max(0, min(bbox.x1, w - 1))
        fy1 = max(0, min(bbox.y1, h - 1))
        fx2 = max(0, min(bbox.x2, w - 1))
        fy2 = max(0, min(bbox.y2, h - 1))

        if fx2 <= fx1 or fy2 <= fy1:
            continue

        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), FACE_COLOR, thickness=2)

        # Check if face is associated with a person track
        assoc_track_id = associated_map.get(id(face))
        if assoc_track_id is not None:
            face_label = f"face -> #{assoc_track_id}"
        else:
            face_label = f"face {face.confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(face_label, font, font_scale, thickness)

        label_y1 = max(fy1 - text_h - 4, 0)
        label_y2 = label_y1 + text_h + 4
        label_x2 = min(fx1 + text_w + 6, w)

        cv2.rectangle(annotated, (fx1, label_y1), (label_x2, label_y2), FACE_COLOR, cv2.FILLED)
        cv2.putText(
            annotated,
            face_label,
            (fx1 + 3, label_y2 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
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
) -> np.ndarray:
    """Draws runtime status HUD on top-left of frame."""
    hud_frame = frame.copy()
    overlay = hud_frame.copy()

    panel_x1, panel_y1 = 15, 15
    panel_x2, panel_y2 = 280, 195

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
    cv2.rectangle(
        hud_frame,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 215, 255),
        1,
    )

    lines = [
        ("VISION - Slice 3 (Face Assoc)", (0, 215, 255), 0.50, 2),
        (
            f"Frame: {current_frame} / {total_frames if total_frames > 0 else 'N/A'}",
            (220, 220, 220),
            0.45,
            1,
        ),
        (f"Source FPS: {source_fps:.1f}", (220, 220, 220), 0.45, 1),
        (
            f"Inference FPS: {inference_fps:.1f}",
            (0, 255, 127) if inference_fps > 0 else (150, 150, 150),
            0.45,
            1,
        ),
        (f"Active Tracks: {active_tracks_count}", (0, 215, 255), 0.45, 1),
        (f"Detections: {detection_count}", (220, 220, 220), 0.45, 1),
        (f"Faces Detected: {faces_detected_count}", (255, 0, 255), 0.45, 1),
        (f"Faces Associated: {faces_associated_count}", (0, 255, 255), 0.45, 1),
    ]

    y_offset = panel_y1 + 20
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
        y_offset += 21

    return hud_frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="VISION Vertical Slice 3 - Face Detection & Person-Track Association"
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
    return parser.parse_args()


def main():
    args = parse_args()

    print("==================================================")
    print("       VISION — Vertical Slice 3 Pipeline        ")
    print("==================================================")
    print(f" Video Path          : {args.video}")
    print(f" YOLO Model          : {args.model}")
    print(f" Confidence Threshold: {args.confidence}")
    print(f" Face Confidence     : {args.face_confidence}")
    print(f" Inference Interval  : Every {args.interval} frame(s)")
    print("==================================================")

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

    window_name = "VISION - Vertical Slice 3 (Face Association)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    latest_detections: List[Detection] = []
    latest_tracks: List[Track] = []
    latest_faces: List[FaceDetection] = []
    latest_associations: List[FaceTrackAssociation] = []

    frame_index = 0
    inference_count = 0
    total_inference_time = 0.0
    recent_inference_fps = 0.0

    total_detections = 0
    total_faces_detected = 0
    total_faces_associated = 0
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

            # Frame Sampling: Run YOLO & Face Detection every Nth frame
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
                    # Safe clamp to frame boundaries
                    px1 = max(0, min(pb.x1, frame_w))
                    py1 = max(0, min(pb.y1, frame_h))
                    px2 = max(0, min(pb.x2, frame_w))
                    py2 = max(0, min(pb.y2, frame_h))

                    if px2 <= px1 or py2 <= py1:
                        continue

                    person_crop = frame[py1:py2, px1:px2]
                    crop_faces = face_detector.detect(person_crop)

                    # Transform crop-relative coordinates to full-frame global coordinates
                    for crop_face in crop_faces:
                        fb = crop_face.bbox
                        gx1 = max(0, min(px1 + fb.x1, frame_w))
                        gy1 = max(0, min(py1 + fb.y1, frame_h))
                        gx2 = max(0, min(px1 + fb.x2, frame_w))
                        gy2 = max(0, min(py1 + fb.y2, frame_h))

                        if gx2 <= gx1 or gy2 <= gy1:
                            continue

                        global_face = FaceDetection(
                            bbox=BoundingBox(x1=gx1, y1=gy1, x2=gx2, y2=gy2),
                            confidence=crop_face.confidence,
                        )
                        current_frame_faces.append(global_face)

                latest_faces = current_frame_faces

                # D. Face-to-Track Association
                latest_associations = associate_faces_to_tracks(latest_tracks, latest_faces)

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

            # Annotate frame
            annotated_frame = draw_annotations(
                frame, latest_tracks, latest_faces, latest_associations
            )

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
            )

            # Display frame
            cv2.imshow(window_name, final_frame)

            target_delay_ms = max(1, int(1000 / source.fps)) if source.fps > 0 else 30
            processing_time_ms = int((time.time() - frame_start) * 1000)
            wait_delay = max(1, target_delay_ms - processing_time_ms)

            key = cv2.waitKey(wait_delay) & 0xFF
            if key == ord("q"):
                print("[INFO] Exit key 'q' pressed by user. Shutting down...")
                break

    finally:
        source.release()
        cv2.destroyAllWindows()

    avg_inf_fps = (
        (inference_count / total_inference_time)
        if total_inference_time > 0
        else 0.0
    )
    print("==================================================")
    print("VISION — Slice 3 Summary")
    print("==================================================")
    print(f"Frames Processed       : {frame_index}")
    print(f"YOLO Inferences        : {inference_count}")
    print(f"Total Detections       : {total_detections}")
    print(f"Unique Tracks          : {len(observed_unique_track_ids)}")
    print(f"Max Active Tracks      : {max_active_tracks}")
    print(f"Faces Detected         : {total_faces_detected}")
    print(f"Faces Associated       : {total_faces_associated}")
    print(f"Average Inference FPS  : {avg_inf_fps:.2f}")
    print("==================================================")


if __name__ == "__main__":
    main()
