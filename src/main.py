import argparse
import os
import sys
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np

from src.core.types import Detection
from src.detection.detector import YOLODetector
from src.ingestion.video import VideoSource

# Color palette for object visualization (BGR format)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "person": (0, 215, 255),     # Bright Amber/Gold
    "car": (0, 255, 127),        # Spring Green
    "truck": (255, 144, 30),     # Dodger Blue
    "bus": (211, 0, 148),        # Deep Purple
    "motorcycle": (255, 255, 0), # Cyan
    "bicycle": (0, 255, 255),    # Bright Yellow
}
DEFAULT_COLOR = (200, 200, 200)


def draw_annotations(
    frame: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:
    """Draws bounding boxes and labels on frame using OpenCV."""
    annotated = frame.copy()

    for det in detections:
        color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)
        bbox = det.bbox

        # Draw bounding box
        cv2.rectangle(
            annotated,
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y2),
            color,
            thickness=2,
        )

        # Label text: e.g. "person 0.91"
        label = f"{det.class_name} {det.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        label_y1 = max(bbox.y1 - text_height - 6, 0)
        label_y2 = label_y1 + text_height + 6
        label_x2 = bbox.x1 + text_width + 8

        # Label background filled rectangle
        cv2.rectangle(
            annotated,
            (bbox.x1, label_y1),
            (label_x2, label_y2),
            color,
            cv2.FILLED,
        )

        # Draw label text
        text_color = (0, 0, 0)
        cv2.putText(
            annotated,
            label,
            (bbox.x1 + 4, label_y2 - baseline - 2),
            font,
            font_scale,
            text_color,
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
    detection_count: int,
) -> np.ndarray:
    """Draws runtime status HUD on top-left of frame."""
    hud_frame = frame.copy()
    overlay = hud_frame.copy()

    panel_x1, panel_y1 = 15, 15
    panel_x2, panel_y2 = 270, 135

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
        ("VISION - Slice 1", (0, 215, 255), 0.55, 2),
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
        (f"Detections: {detection_count}", (255, 255, 255), 0.45, 1),
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
        y_offset += 22

    return hud_frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="VISION Vertical Slice 1 - Real-Time Video Object Detection"
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
        help="Confidence threshold for YOLO detection (default: 0.25)",
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
    print("       VISION — Vertical Slice 1 Pipeline        ")
    print("==================================================")
    print(f" Video Path          : {args.video}")
    print(f" Model               : {args.model}")
    print(f" Confidence Threshold: {args.confidence}")
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

    # 2. Detector Initialization
    try:
        detector = YOLODetector(
            model_name=args.model,
            confidence_threshold=args.confidence,
        )
    except Exception as e:
        print(f"[ERROR] Detector initialization failed: {e}", file=sys.stderr)
        source.release()
        sys.exit(1)

    window_name = "VISION - Vertical Slice 1"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    latest_detections: List[Detection] = []
    frame_index = 0
    inference_count = 0
    total_inference_time = 0.0
    recent_inference_fps = 0.0
    total_detections = 0

    target_class_order = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
    class_counts: Dict[str, int] = {cls_name: 0 for cls_name in target_class_order}

    start_time = time.time()

    try:
        while True:
            frame_start = time.time()
            frame = source.read_frame()

            if frame is None:
                print("[INFO] Reached end of video stream.")
                break

            frame_index = source.current_frame

            # Frame Sampling: Run YOLO inference every Nth frame
            if (frame_index - 1) % args.interval == 0:
                t0 = time.time()
                latest_detections = detector.detect(frame)
                t1 = time.time()

                inf_time = t1 - t0
                total_inference_time += inf_time
                inference_count += 1
                if inf_time > 0:
                    recent_inference_fps = 1.0 / inf_time

                total_detections += len(latest_detections)
                for det in latest_detections:
                    if det.class_name in class_counts:
                        class_counts[det.class_name] += 1
                    else:
                        class_counts[det.class_name] = 1

            elapsed_total = time.time() - start_time
            actual_source_fps = (
                frame_index / elapsed_total if elapsed_total > 0 else source.fps
            )

            # Annotate frame
            annotated_frame = draw_annotations(frame, latest_detections)

            # Draw HUD
            final_frame = draw_hud(
                frame=annotated_frame,
                current_frame=frame_index,
                total_frames=source.frame_count,
                source_fps=actual_source_fps,
                inference_fps=recent_inference_fps,
                detection_count=len(latest_detections),
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
    print("VISION — Detection Summary")
    print("==================================================")
    print(f"Frames Processed : {frame_index}")
    print(f"YOLO Inferences  : {inference_count}")
    print(f"Total Detections : {total_detections}")
    print("")
    print("Detection Classes:")
    for cls_name in target_class_order:
        count = class_counts.get(cls_name, 0)
        print(f"  {cls_name:<10} : {count}")
    print("")
    print(f"Average Inference FPS : {avg_inf_fps:.2f}")
    print("==================================================")


if __name__ == "__main__":
    main()
