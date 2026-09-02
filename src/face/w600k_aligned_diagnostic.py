import argparse
import os
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple
import cv2
import numpy as np

from src.face.alignment import align_face
from src.face.detector import FaceDetector
from src.face.gallery import SUPPORTED_IMAGE_EXTENSIONS
from src.face.modern_embedder import W600KR50Embedder


class AlignedStat(NamedTuple):
    identity: str
    filename: str
    unaligned_vec: np.ndarray
    aligned_vec: np.ndarray


def evaluate_w600k_alignment(
    gallery_dir: str,
    embedder: W600KR50Embedder,
    detector: FaceDetector,
) -> Tuple[List[AlignedStat], np.ndarray, np.ndarray, List[float], List[float], List[float], List[float]]:
    """Extracts both unaligned and aligned embeddings across all gallery images."""
    identity_dirs = sorted([
        d for d in os.listdir(gallery_dir)
        if os.path.isdir(os.path.join(gallery_dir, d))
    ])

    stats: List[AlignedStat] = []

    for identity in identity_dirs:
        id_dir = os.path.join(gallery_dir, identity)
        for entry in sorted(os.listdir(id_dir)):
            entry_path = os.path.join(id_dir, entry)
            if not os.path.isfile(entry_path):
                continue
            if os.path.splitext(entry)[1].lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            img = cv2.imread(entry_path)
            if img is None or img.size == 0:
                continue

            faces = detector.detect(img)
            if not faces:
                continue

            best_face = max(faces, key=lambda f: f.confidence)
            fb = best_face.bbox
            fh, fw = img.shape[:2]

            fx1 = max(0, min(fb.x1, fw))
            fy1 = max(0, min(fb.y1, fh))
            fx2 = max(0, min(fb.x2, fw))
            fy2 = max(0, min(fb.y2, fh))

            if fx2 <= fx1 or fy2 <= fy1:
                continue

            # 1. Unaligned crop
            unaligned_crop = img[fy1:fy2, fx1:fx2]
            emb_unaligned = embedder.embed(unaligned_crop)

            # 2. Aligned crop
            aligned_crop = align_face(img, best_face.landmarks)
            emb_aligned = embedder.embed(aligned_crop) if aligned_crop is not None else None

            if emb_unaligned is not None and emb_aligned is not None:
                stats.append(
                    AlignedStat(
                        identity=identity,
                        filename=entry,
                        unaligned_vec=emb_unaligned.vector,
                        aligned_vec=emb_aligned.vector,
                    )
                )

    N = len(stats)
    unaligned_matrix = np.zeros((N, N), dtype=np.float32)
    aligned_matrix = np.zeros((N, N), dtype=np.float32)

    unaligned_same: List[float] = []
    unaligned_cross: List[float] = []
    aligned_same: List[float] = []
    aligned_cross: List[float] = []

    for i in range(N):
        for j in range(N):
            sim_u = float(np.dot(stats[i].unaligned_vec, stats[j].unaligned_vec))
            sim_a = float(np.dot(stats[i].aligned_vec, stats[j].aligned_vec))
            unaligned_matrix[i, j] = sim_u
            aligned_matrix[i, j] = sim_a

            if i < j:
                if stats[i].identity == stats[j].identity:
                    unaligned_same.append(sim_u)
                    aligned_same.append(sim_a)
                else:
                    unaligned_cross.append(sim_u)
                    aligned_cross.append(sim_a)

    return (
        stats,
        unaligned_matrix,
        aligned_matrix,
        unaligned_same,
        unaligned_cross,
        aligned_same,
        aligned_cross,
    )


def print_matrix(matrix: np.ndarray, labels: List[str], title: str) -> None:
    """Formats and prints cosine similarity matrix."""
    print(f"\n==================================================")
    print(f" {title} ")
    print("==================================================")
    header = f"{'':12s}" + "".join([f"{lbl:11s}" for lbl in labels])
    print(header)
    print("-" * len(header))
    for i in range(len(labels)):
        row_str = f"{labels[i]:12s}" + "".join([f"{matrix[i, j]:11.4f}" for j in range(len(labels))])
        print(row_str)


def main():
    parser = argparse.ArgumentParser(
        description="VISION Slice 5.6 - W600K-R50 Landmark-Aligned vs Unaligned Diagnostic"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/w600k_r50.onnx",
        help="Path to InsightFace w600k_r50 model",
    )
    parser.add_argument(
        "--gallery-dir",
        type=str,
        default="data/face_gallery",
        help="Path to gallery directory",
    )
    args = parser.parse_args()

    print("==================================================")
    print("  W600K-R50 LANDMARK ALIGNMENT BENCHMARK          ")
    print("==================================================")
    print(f" Model Path  : {args.model_path}")
    print(f" Gallery Dir : {args.gallery_dir}")

    detector = FaceDetector(score_threshold=0.50)
    embedder = W600KR50Embedder(model_path=args.model_path)

    (
        stats,
        unaligned_mat,
        aligned_mat,
        unaligned_same,
        unaligned_cross,
        aligned_same,
        aligned_cross,
    ) = evaluate_w600k_alignment(args.gallery_dir, embedder, detector)

    labels = [f"{s.identity[:5]}_{s.filename[:4]}" for s in stats]

    # Print matrices
    print_matrix(unaligned_mat, labels, "A. UNALIGNED W600K-R50 SIMILARITY MATRIX")
    print_matrix(aligned_mat, labels, "B. LANDMARK-ALIGNED W600K-R50 SIMILARITY MATRIX")

    # Metrics computation
    u_mean_g, u_min_g, u_max_g = float(np.mean(unaligned_same)), float(np.min(unaligned_same)), float(np.max(unaligned_same))
    u_mean_i, u_min_i, u_max_i = float(np.mean(unaligned_cross)), float(np.min(unaligned_cross)), float(np.max(unaligned_cross))
    u_sep = u_mean_g - u_mean_i

    a_mean_g, a_min_g, a_max_g = float(np.mean(aligned_same)), float(np.min(aligned_same)), float(np.max(aligned_same))
    a_mean_i, a_min_i, a_max_i = float(np.mean(aligned_cross)), float(np.min(aligned_cross)), float(np.max(aligned_cross))
    a_sep = a_mean_g - a_mean_i

    print("\n--------------------------------------------------")
    print("A. UNALIGNED W600K-R50 BENCHMARK METRICS")
    print("--------------------------------------------------")
    print(f" Genuine Mean : {u_mean_g:.4f} (Min: {u_min_g:.4f}, Max: {u_max_g:.4f})")
    print(f" Impostor Mean: {u_mean_i:.4f} (Min: {u_min_i:.4f}, Max: {u_max_i:.4f})")
    print(f" Separation   : {u_sep:.4f}")

    print("\n--------------------------------------------------")
    print("B. LANDMARK-ALIGNED W600K-R50 BENCHMARK METRICS")
    print("--------------------------------------------------")
    print(f" Genuine Mean : {a_mean_g:.4f} (Min: {a_min_g:.4f}, Max: {a_max_g:.4f})")
    print(f" Impostor Mean: {a_mean_i:.4f} (Min: {a_min_i:.4f}, Max: {a_max_i:.4f})")
    print(f" Separation   : {a_sep:.4f} (Genuine Mean - Impostor Mean)")

    # Specific pairs
    print("\n--------------------------------------------------")
    print("KEY PAIR COMPARISONS (UNALIGNED vs ALIGNED)")
    print("--------------------------------------------------")
    
    # Locate indices
    idx_map = {f"{s.identity}_{s.filename}": i for i, s in enumerate(stats)}
    
    def report_pair(id1_file: str, id2_file: str, label: str):
        if id1_file in idx_map and id2_file in idx_map:
            i, j = idx_map[id1_file], idx_map[id2_file]
            sim_u = unaligned_mat[i, j]
            sim_a = aligned_mat[i, j]
            print(f" {label:35s} | Unaligned: {sim_u:7.4f} | Aligned: {sim_a:7.4f}")

    report_pair("Shreyas_Chavan_left.jpeg", "Shreyas_Chavan_right.jpeg", "Shreyas left <-> right")
    report_pair("Shreyas_Chavan_front.jpeg", "Shreyas_Chavan_left.jpeg", "Shreyas front <-> left")
    report_pair("Shreyas_Chavan_front.jpeg", "Shreyas_Chavan_right.jpeg", "Shreyas front <-> right")
    report_pair("Atharva_Jaysingpure_left.jpeg", "Atharva_Jaysingpure_right.jpeg", "Atharva left <-> right")
    report_pair("Atharva_Jaysingpure_front.jpeg", "Atharva_Jaysingpure_left.jpeg", "Atharva front <-> left")
    report_pair("Atharva_Jaysingpure_front.jpeg", "Atharva_Jaysingpure_right.jpeg", "Atharva front <-> right")
    report_pair("Atharva_Jaysingpure_front.jpeg", "Shreyas_Chavan_front.jpeg", "Atharva front <-> Shreyas front")

    print("\n==================================================")
    print("                    VERDICT                       ")
    print("==================================================")
    if a_sep > 0.40 and a_min_g > 0.50 and a_max_i < 0.20:
        print("PASS — Landmark alignment creates clear discrimination between genuine and impostor identities.")
    else:
        print("WARNING — Further landmark or preprocessing investigation required.")
    print("==================================================")


if __name__ == "__main__":
    main()
