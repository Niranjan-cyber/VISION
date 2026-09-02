import argparse
import os
import sys
import time
from typing import Dict, List, Set, Tuple
import cv2
import numpy as np

from src.core.types import (
    BoundingBox,
    Detection,
    FaceDetection,
    FaceEmbedding,
    IdentityMatch,
    Track,
)
from src.detection.detector import YOLODetector
from src.face.association import FaceTrackAssociation, associate_faces_to_tracks
from src.face.detector import FaceDetector
from src.face.embedder import FaceEmbedder
from src.face.gallery import load_gallery_from_dir
from src.face.matcher import FaceMatcher
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
FACE_KNOWN_COLOR = (0, 255, 127)   # Spring Green for recognized identity match
FACE_UNKNOWN_COLOR = (255, 0, 255) # Magenta for unknown / unassociated face
DEFAULT_COLOR = (200, 200, 200)


def draw_annotations(
    frame: np.ndarray,
    tracks: List[Track],
    faces: List[FaceDetection],
    associations: List[FaceTrackAssociation],
    track_identity_map: Dict[int, IdentityMatch],
) -> np.ndarray:
    """Renders tracked object bounding boxes, face boxes, and identity match status indicators."""
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

        x1 = max(0, min(bbox.x1, w - 1))
        y1 = max(0, min(bbox.y1, h - 1))
        x2 = max(0, min(bbox.x2, w - 1))
        y2 = max(0, min(bbox.y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

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

    # 2. Render Face Boxes & Identity Match Labels
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
            box_color = FACE_UNKNOWN_COLOR
            sim_str = f" ({match_info.similarity:.2f})" if match_info is not None else ""
            face_label = f"face -> #{assoc_track_id} | Unknown{sim_str}"
        else:
            box_color = FACE_UNKNOWN_COLOR
            face_label = f"face {face.confidence:.2f}"

        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), box_color, thickness=2)

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
    unknown_faces_count: int,
    recog_threshold: float,
    recog_margin: float,
) -> np.ndarray:
    """Draws runtime status HUD on top-left of frame."""
    hud_frame = frame.copy()
    overlay = hud_frame.copy()

    panel_x1, panel_y1 = 15, 15
    panel_x2, panel_y2 = 295, 295

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
        ("VISION - Slice 5 (Face Recog)", (0, 215, 255), 0.48, 2),
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
        (f"Unknown Faces: {unknown_faces_count}", (255, 0, 255), 0.42, 1),
        (f"Recog Threshold: {recog_threshold:.2f}", (200, 200, 200), 0.40, 1),
        (f"Recog Margin: {recog_margin:.2f}", (200, 200, 200), 0.40, 1),
    ]

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
        y_offset += 21

    return hud_frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="VISION Vertical Slice 5 - Face Recognition & Identity Matching Pipeline"
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
        "--debug-face-matching",
        action="store_true",
        help="Print diagnostic log of candidate similarity scores for newly evaluated track embeddings",
    )
    parser.add_argument(
        "--debug-face-crops",
        action="store_true",
        help="Save extracted face crop images to scratch/debug_face_crops/ for visual inspection",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("==================================================")
    print("       VISION — Vertical Slice 5 Pipeline        ")
    print("==================================================")
    print(f" Video Path          : {args.video}")
    print(f" YOLO Model          : {args.model}")
    print(f" Confidence Threshold: {args.confidence}")
    print(f" Face Confidence     : {args.face_confidence}")
    print(f" Recognition Thresh  : {args.face_threshold:.2f}")
    print(f" Recognition Margin  : {args.face_margin:.2f}")
    print(f" Gallery Directory   : {args.gallery_dir}")
    print(f" Debug Matching      : {args.debug_face_matching}")
    print(f" Debug Face Crops    : {args.debug_face_crops}")
    print(f" Inference Interval  : Every {args.interval} frame(s)")
    print("==================================================")

    if args.debug_face_crops:
        os.makedirs("scratch/debug_face_crops", exist_ok=True)

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

    # 5. FaceEmbedder Initialization
    try:
        face_embedder = FaceEmbedder()
    except Exception as e:
        print(f"[ERROR] FaceEmbedder initialization failed: {e}", file=sys.stderr)
        source.release()
        sys.exit(1)

    # 6. FaceGallery & FaceMatcher Initialization
    try:
        gallery = load_gallery_from_dir(
            args.gallery_dir, face_detector, face_embedder
        )
        face_matcher = FaceMatcher(
            gallery, threshold=args.face_threshold, margin=args.face_margin
        )
    except Exception as e:
        print(f"[ERROR] Face Gallery/Matcher initialization failed: {e}", file=sys.stderr)
        source.release()
        sys.exit(1)

    window_name = "VISION - Vertical Slice 5 (Face Recognition)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    latest_detections: List[Detection] = []
    latest_tracks: List[Track] = []
    latest_faces: List[FaceDetection] = []
    latest_associations: List[FaceTrackAssociation] = []

    # Track-level identity cache: track_id -> IdentityMatch
    track_identity_cache: Dict[int, IdentityMatch] = {}

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

                        global_face = FaceDetection(
                            bbox=BoundingBox(x1=gx1, y1=gy1, x2=gx2, y2=gy2),
                            confidence=crop_face.confidence,
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
                        fb = assoc.face.bbox
                        fx1 = max(0, min(fb.x1, frame_w))
                        fy1 = max(0, min(fb.y1, frame_h))
                        fx2 = max(0, min(fb.x2, frame_w))
                        fy2 = max(0, min(fb.y2, frame_h))

                        if fx2 <= fx1 or fy2 <= fy1:
                            continue

                        face_crop = frame[fy1:fy2, fx1:fx2]
                        if face_crop.size == 0:
                            continue

                        if args.debug_face_crops:
                            crop_save_path = f"scratch/debug_face_crops/track_{track_id}_frame_{frame_index}.jpg"
                            cv2.imwrite(crop_save_path, face_crop)
                            print(f"[DEBUG] Saved face crop ({face_crop.shape[1]}x{face_crop.shape[0]}) to '{crop_save_path}'")

                        embedding = face_embedder.embed(face_crop)
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
                        else:
                            match_result = IdentityMatch(identity=None, similarity=0.0, is_match=False)

                    if match_result.is_match:
                        frame_recognized_count += 1
                    else:
                        frame_unknown_count += 1

                total_recognized_faces += frame_recognized_count
                total_unknown_faces += frame_unknown_count

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
                frame,
                latest_tracks,
                latest_faces,
                latest_associations,
                track_identity_cache,
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
                embeddings_generated_count=total_embeddings_generated,
                recognized_faces_count=total_recognized_faces,
                unknown_faces_count=total_unknown_faces,
                recog_threshold=args.face_threshold,
                recog_margin=args.face_margin,
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
    print("VISION — Slice 5 Summary")
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
    print(f"Recognition Threshold  : {args.face_threshold:.2f}")
    print(f"Recognition Margin     : {args.face_margin:.2f}")
    print(f"Average Inference FPS  : {avg_inf_fps:.2f}")
    print("==================================================")


if __name__ == "__main__":
    main()
