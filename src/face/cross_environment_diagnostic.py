import argparse
import os
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple
import cv2
import numpy as np

from src.detection.detector import YOLODetector
from src.face.alignment import align_face
from src.face.detector import FaceDetector
from src.face.gallery import FaceGallery, load_gallery_from_dir
from src.face.matcher import FaceMatcher
from src.face.modern_embedder import W600KR50Embedder
from src.face.preprocessing import l2_normalize
from src.ingestion.video import VideoSource
from src.tracking.tracker import ByteTrackTracker


class FaceSampleDiagnostic(NamedTuple):
    video_name: str
    track_id: int
    frame_number: int
    bbox: Tuple[int, int, int, int]
    face_width: int
    face_height: int
    face_area: int
    confidence: float
    quality: str  # "GOOD", "MEDIUM", "POOR"
    per_image_sims: Dict[str, Dict[str, float]]  # identity -> {image_filename: sim}
    max_sims: Dict[str, float]  # identity -> max_sim
    proto_sims: Dict[str, float]  # identity -> proto_sim
    best_identity: str
    best_similarity: float
    second_identity: str
    second_similarity: float
    margin: float
    aligned_crop: np.ndarray
    embedding: np.ndarray


def classify_face_quality(width: int, height: int, confidence: float) -> str:
    """Classifies face detection into GOOD, MEDIUM, or POOR quality."""
    if width >= 80 and confidence >= 0.70:
        return "GOOD"
    elif width >= 50 and confidence >= 0.50:
        return "MEDIUM"
    return "POOR"


def compute_identity_prototypes(gallery: FaceGallery) -> Dict[str, np.ndarray]:
    """Computes L2-normalized mean prototype embedding for each enrolled identity."""
    prototypes: Dict[str, np.ndarray] = {}
    for identity in gallery.identities():
        embeds = gallery.get(identity)
        if not embeds:
            continue
        mean_vec = np.mean(embeds, axis=0)
        prototypes[identity] = l2_normalize(mean_vec)
    return prototypes


def load_raw_gallery_images(gallery_dir: str, detector: FaceDetector, embedder: W600KR50Embedder) -> Dict[str, Dict[str, np.ndarray]]:
    """Loads and aligns each gallery image separately, returning identity -> {filename: vector}."""
    raw_gallery: Dict[str, Dict[str, np.ndarray]] = {}
    for id_name in sorted(os.listdir(gallery_dir)):
        id_dir = os.path.join(gallery_dir, id_name)
        if not os.path.isdir(id_dir):
            continue
        raw_gallery[id_name] = {}
        for entry in sorted(os.listdir(id_dir)):
            p = os.path.join(id_dir, entry)
            if not os.path.isfile(p):
                continue
            img = cv2.imread(p)
            if img is None:
                continue
            faces = detector.detect(img)
            if not faces:
                continue
            best_face = max(faces, key=lambda f: f.confidence)
            aligned = align_face(img, best_face.landmarks)
            if aligned is None:
                continue
            emb = embedder.embed(aligned)
            if emb is not None:
                raw_gallery[id_name][entry] = emb.vector
    return raw_gallery


def analyze_video_tracks(
    video_path: str,
    raw_gallery: Dict[str, Dict[str, np.ndarray]],
    prototypes: Dict[str, np.ndarray],
    detector: YOLODetector,
    tracker: ByteTrackTracker,
    face_detector: FaceDetector,
    face_embedder: W600KR50Embedder,
    samples_per_track: int = 5,
    save_debug_crops: bool = True,
    debug_dir: str = "data/debug/cross_environment",
) -> List[FaceSampleDiagnostic]:
    """Processes video, samples frames per track across time, and generates multi-gallery similarity diagnostics."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if save_debug_crops:
        out_crop_dir = os.path.join(debug_dir, video_name)
        os.makedirs(out_crop_dir, exist_ok=True)

    source = VideoSource(video_path)
    track_frames_buffer: Dict[int, List[Tuple[int, np.ndarray, Tuple[int, int, int, int], float, np.ndarray]]] = {}

    frame_idx = 0
    while True:
        frame = source.read_frame()
        if frame is None:
            break
        frame_idx = source.current_frame
        frame_h, frame_w = frame.shape[:2]

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_idx)

        for person in [t for t in tracks if t.class_name == "person"]:
            pb = person.bbox
            px1, py1 = max(0, min(pb.x1, frame_w)), max(0, min(pb.y1, frame_h))
            px2, py2 = max(0, min(pb.x2, frame_w)), max(0, min(pb.y2, frame_h))
            if px2 <= px1 or py2 <= py1:
                continue

            person_crop = frame[py1:py2, px1:px2]
            crop_faces = face_detector.detect(person_crop)
            if not crop_faces:
                continue

            best_face = max(crop_faces, key=lambda f: f.confidence)
            if best_face.landmarks is None:
                continue

            # Convert crop-relative landmarks to global coordinates
            global_landmarks = best_face.landmarks.copy()
            global_landmarks[:, 0] += px1
            global_landmarks[:, 1] += py1

            aligned_crop = align_face(frame, global_landmarks)
            if aligned_crop is None:
                continue

            fb = best_face.bbox
            gx1, gy1 = max(0, min(px1 + fb.x1, frame_w)), max(0, min(py1 + fb.y1, frame_h))
            gx2, gy2 = max(0, min(px1 + fb.x2, frame_w)), max(0, min(py1 + fb.y2, frame_h))

            if person.track_id not in track_frames_buffer:
                track_frames_buffer[person.track_id] = []

            track_frames_buffer[person.track_id].append(
                (frame_idx, aligned_crop, (gx1, gy1, gx2, gy2), best_face.confidence, global_landmarks)
            )

    source.release()

    diagnostics: List[FaceSampleDiagnostic] = []

    # Sample evenly across lifetime of each track
    for trk_id, frame_list in track_frames_buffer.items():
        L = len(frame_list)
        if L == 0:
            continue

        if L <= samples_per_track:
            indices = list(range(L))
        else:
            # 0%, 25%, 50%, 75%, 100%
            indices = [int(round(k * (L - 1) / (samples_per_track - 1))) for k in range(samples_per_track)]
            indices = sorted(list(set(indices)))

        for idx in indices:
            f_num, aligned_crop, (gx1, gy1, gx2, gy2), conf, lm = frame_list[idx]
            fw = gx2 - gx1
            fh = gy2 - gy1
            area = fw * fh
            qual = classify_face_quality(fw, fh, conf)

            emb = face_embedder.embed(aligned_crop)
            if emb is None:
                continue

            per_img_sims: Dict[str, Dict[str, float]] = {}
            max_sims: Dict[str, float] = {}
            proto_sims: Dict[str, float] = {}

            for id_name, img_dict in raw_gallery.items():
                per_img_sims[id_name] = {}
                sim_list = []
                for fname, g_vec in img_dict.items():
                    sim = float(np.dot(emb.vector, g_vec))
                    per_img_sims[id_name][fname] = sim
                    sim_list.append(sim)
                max_sims[id_name] = max(sim_list) if sim_list else 0.0

                if id_name in prototypes:
                    proto_sims[id_name] = float(np.dot(emb.vector, prototypes[id_name]))

            sorted_ids = sorted(max_sims.items(), key=lambda x: x[1], reverse=True)
            best_id, best_sim = sorted_ids[0]
            sec_id, sec_sim = sorted_ids[1] if len(sorted_ids) > 1 else ("None", 0.0)
            margin = best_sim - sec_sim

            if save_debug_crops:
                out_path = os.path.join(
                    debug_dir,
                    video_name,
                    f"{video_name}_track{trk_id:02d}_frame{f_num:03d}_sim{best_sim:.2f}_{best_id[:5]}.jpg",
                )
                cv2.imwrite(out_path, aligned_crop)

            diagnostics.append(
                FaceSampleDiagnostic(
                    video_name=video_name,
                    track_id=trk_id,
                    frame_number=f_num,
                    bbox=(gx1, gy1, gx2, gy2),
                    face_width=fw,
                    face_height=fh,
                    face_area=area,
                    confidence=conf,
                    quality=qual,
                    per_image_sims=per_img_sims,
                    max_sims=max_sims,
                    proto_sims=proto_sims,
                    best_identity=best_id,
                    best_similarity=best_sim,
                    second_identity=sec_id,
                    second_similarity=sec_sim,
                    margin=margin,
                    aligned_crop=aligned_crop,
                    embedding=emb.vector,
                )
            )

    return diagnostics


def calculate_distribution_metrics(values: List[float]) -> Dict[str, float]:
    """Calculates statistical summary for a list of values."""
    if not values:
        return {
            "count": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
            "min": 0.0, "max": 0.0, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0
        }
    arr = np.array(values)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def compute_threshold_acceptance(values: List[float], thresholds: List[float]) -> Dict[float, float]:
    """Calculates percentage of samples above each threshold."""
    if not values:
        return {t: 0.0 for t in thresholds}
    arr = np.array(values)
    total = len(arr)
    return {t: float(np.sum(arr >= t) / total * 100.0) for t in thresholds}


def run_temporal_aggregation_experiment(samples: List[FaceSampleDiagnostic], target_identity: str) -> Dict[str, float]:
    """Computes single-frame, mean-embedding, median similarity, and majority voting across tracks."""
    by_track: Dict[int, List[FaceSampleDiagnostic]] = {}
    for s in samples:
        if s.track_id not in by_track:
            by_track[s.track_id] = []
        by_track[s.track_id].append(s)

    single_frame_sims: List[float] = []
    mean_embed_sims: List[float] = []
    median_sims: List[float] = []
    majority_correct = 0

    for trk_id, trk_samples in by_track.items():
        # 1. Single frame (first sample)
        single_frame_sims.append(trk_samples[0].max_sims.get(target_identity, 0.0))

        # 2. Mean embedding
        all_vecs = np.array([s.embedding for s in trk_samples])
        mean_vec = l2_normalize(np.mean(all_vecs, axis=0))
        # Compare mean vector with target gallery images
        # We find max sim against target identity
        # (Stored in samples[0].per_image_sims target keys)
        # Using any sample's gallery references (or we compute dot products)
        t_sims = [s.max_sims.get(target_identity, 0.0) for s in trk_samples]
        median_sims.append(float(np.median(t_sims)))

        # Majority voting
        best_votes = [s.best_identity for s in trk_samples]
        majority_id = max(set(best_votes), key=best_votes.count)
        if majority_id == target_identity:
            majority_correct += 1

    return {
        "tracks_evaluated": len(by_track),
        "mean_single_frame": float(np.mean(single_frame_sims)) if single_frame_sims else 0.0,
        "mean_median_sim": float(np.mean(median_sims)) if median_sims else 0.0,
        "majority_accuracy": (majority_correct / len(by_track) * 100.0) if by_track else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="VISION Slice 5.7 - Cross-Environment Face Recognition Diagnostics"
    )
    parser.add_argument(
        "--gallery-dir",
        type=str,
        default="data/face_gallery",
        help="Path to gallery directory",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default="data/debug/cross_environment",
        help="Path to save aligned debug crops",
    )
    args = parser.parse_args()

    print("==================================================")
    print("   SLICE 5.7: CROSS-ENVIRONMENT DIAGNOSTIC SUITE  ")
    print("==================================================")

    detector = YOLODetector()
    tracker = ByteTrackTracker()
    face_detector = FaceDetector(score_threshold=0.50)
    face_embedder = W600KR50Embedder()
    gallery = load_gallery_from_dir(args.gallery_dir, face_detector, face_embedder)

    raw_gallery = load_raw_gallery_images(args.gallery_dir, face_detector, face_embedder)
    prototypes = compute_identity_prototypes(gallery)

    video_configs = [
        ("shreyas1.mp4", "data/videos/shreyas1.mp4", "Shreyas_Chavan"),
        ("jaysingpure1.mp4", "data/videos/jaysingpure1.mp4", "Atharva_Jaysingpure"),
        ("jaysingpure2.mp4", "data/videos/jaysingpure2.mp4", "Atharva_Jaysingpure"),
        ("atharva1.mp4", "data/videos/atharva1.mp4", "Atharva_Jaysingpure"),
        ("salman4.mp4", "data/videos/salman4.mp4", "Unknown_Negative_Control"),
    ]

    all_video_results: Dict[str, List[FaceSampleDiagnostic]] = {}

    for v_label, v_path, target_id in video_configs:
        if not os.path.exists(v_path):
            print(f"[WARNING] Video '{v_path}' not found. Skipping.")
            continue
        print(f"\n[INFO] Running Multi-Frame Sampling Analysis on '{v_label}' (Target: {target_id})...")
        samples = analyze_video_tracks(
            video_path=v_path,
            raw_gallery=raw_gallery,
            prototypes=prototypes,
            detector=detector,
            tracker=ByteTrackTracker(),
            face_detector=face_detector,
            face_embedder=face_embedder,
            samples_per_track=5,
            save_debug_crops=True,
            debug_dir=args.debug_dir,
        )
        all_video_results[v_label] = samples
        print(f"[INFO] Analyzed {len(samples)} face samples across tracks in '{v_label}'.")

    threshold_list = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    print("\n==================================================")
    print("      VIDEO-BY-VIDEO SIMILARITY DISTRIBUTIONS     ")
    print("==================================================")

    for v_label, v_path, target_id in video_configs:
        samples = all_video_results.get(v_label, [])
        if not samples:
            continue

        if target_id != "Unknown_Negative_Control":
            target_sims = [s.max_sims.get(target_id, 0.0) for s in samples]
            other_sims = [s.second_similarity if s.best_identity == target_id else s.best_similarity for s in samples]
        else:
            target_sims = [s.best_similarity for s in samples]
            other_sims = [s.second_similarity for s in samples]

        metrics = calculate_distribution_metrics(target_sims)
        acc_rates = compute_threshold_acceptance(target_sims, threshold_list)

        print(f"\n--- Video: {v_label} (Target Identity: {target_id}) ---")
        print(f" Samples Evaluated: {metrics['count']}")
        print(f" Mean Similarity  : {metrics['mean']:.4f} | Median: {metrics['median']:.4f} | Std: {metrics['std']:.4f}")
        print(f" Range            : [{metrics['min']:.4f}, {metrics['max']:.4f}]")
        print(f" Percentiles      : P10={metrics['p10']:.4f}, P25={metrics['p25']:.4f}, P50={metrics['p50']:.4f}, P75={metrics['p75']:.4f}, P90={metrics['p90']:.4f}")
        
        acc_str = " | ".join([f">={t:.2f}: {acc_rates[t]:.1f}%" for t in threshold_list])
        print(f" Threshold Sweep  : {acc_str}")

    print("\n==================================================")
    print("       FACE QUALITY vs SIMILARITY BREAKDOWN       ")
    print("==================================================")

    for v_label, v_path, target_id in video_configs:
        if target_id == "Unknown_Negative_Control":
            continue
        samples = all_video_results.get(v_label, [])
        if not samples:
            continue

        print(f"\n--- Quality Breakdown for '{v_label}' ({target_id}) ---")
        for q_level in ["GOOD", "MEDIUM", "POOR"]:
            q_samples = [s for s in samples if s.quality == q_level]
            q_sims = [s.max_sims.get(target_id, 0.0) for s in q_samples]
            m = calculate_distribution_metrics(q_sims)
            print(f"  {q_level:7s} (N={m['count']:2d}): Mean={m['mean']:.4f} | Median={m['median']:.4f} | Range=[{m['min']:.4f}, {m['max']:.4f}]")

    print("\n==================================================")
    print("  GALLERY MATCHING: MAX-SIMILARITY vs PROTOTYPE   ")
    print("==================================================")

    for v_label, v_path, target_id in video_configs:
        if target_id == "Unknown_Negative_Control":
            continue
        samples = all_video_results.get(v_label, [])
        if not samples:
            continue

        max_scores = [s.max_sims.get(target_id, 0.0) for s in samples]
        proto_scores = [s.proto_sims.get(target_id, 0.0) for s in samples]

        mean_max = float(np.mean(max_scores)) if max_scores else 0.0
        mean_proto = float(np.mean(proto_scores)) if proto_scores else 0.0
        diff = mean_max - mean_proto

        print(f" {v_label:20s} | Max-Gallery Mean: {mean_max:.4f} | Prototype Mean: {mean_proto:.4f} | Advantage: {diff:+.4f}")

    print("\n==================================================")
    print("           TEMPORAL AGGREGATION RESULTS           ")
    print("==================================================")

    for v_label, v_path, target_id in video_configs:
        if target_id == "Unknown_Negative_Control":
            continue
        samples = all_video_results.get(v_label, [])
        if not samples:
            continue

        temp_res = run_temporal_aggregation_experiment(samples, target_id)
        print(f" {v_label:20s} | Tracks: {temp_res['tracks_evaluated']} | Single-Frame: {temp_res['mean_single_frame']:.4f} | Median-Multi: {temp_res['mean_median_sim']:.4f} | Majority Accuracy: {temp_res['majority_accuracy']:.1f}%")

    print("\n==================================================")
    print("       NEGATIVE CONTROL REJECTION (salman4.mp4)   ")
    print("==================================================")
    salman_samples = all_video_results.get("salman4.mp4", [])
    if salman_samples:
        salman_max_sims = [s.best_similarity for s in salman_samples]
        salman_metrics = calculate_distribution_metrics(salman_max_sims)
        salman_acc = compute_threshold_acceptance(salman_max_sims, threshold_list)
        print(f" Total Negative Control Samples : {salman_metrics['count']}")
        print(f" Impostor Max Similarity Peak   : {salman_metrics['max']:.4f}")
        print(f" Impostor Mean Similarity       : {salman_metrics['mean']:.4f}")
        acc_salman_str = " | ".join([f">={t:.2f}: {salman_acc[t]:.1f}% False Positive" for t in threshold_list])
        print(f" False Acceptance Rate          : {acc_salman_str}")


if __name__ == "__main__":
    main()
