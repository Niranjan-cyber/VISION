import argparse
import os
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple
import cv2
import numpy as np

from src.face.detector import FaceDetector
from src.face.embedder import ONNXRuntimeArcFaceEmbedder, OpenCVArcFaceEmbedder
from src.face.gallery import SUPPORTED_IMAGE_EXTENSIONS


class ImageEmbeddingStat(NamedTuple):
    identity: str
    filename: str
    backend: str
    dimension: int
    norm: float
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    vector: np.ndarray


def extract_face_crop(img: np.ndarray, detector: FaceDetector) -> Optional[np.ndarray]:
    """Detects best face in image and returns cropped face region."""
    if img is None or img.size == 0:
        return None
    faces = detector.detect(img)
    if not faces:
        return None
    best_face = max(faces, key=lambda f: f.confidence)
    fb = best_face.bbox
    h, w = img.shape[:2]
    x1, y1 = max(0, min(fb.x1, w)), max(0, min(fb.y1, h))
    x2, y2 = min(w, max(x1, fb.x2)), min(h, max(y1, fb.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def evaluate_backend(
    gallery_dir: str,
    embedder,
    backend_name: str,
    detector: FaceDetector,
) -> Tuple[List[ImageEmbeddingStat], np.ndarray, List[float], List[float]]:
    """Runs embedding extraction across gallery images for a specific backend."""
    identity_dirs = sorted([
        d for d in os.listdir(gallery_dir)
        if os.path.isdir(os.path.join(gallery_dir, d))
    ])

    stats: List[ImageEmbeddingStat] = []

    for identity in identity_dirs:
        id_dir = os.path.join(gallery_dir, identity)
        for entry in sorted(os.listdir(id_dir)):
            entry_path = os.path.join(id_dir, entry)
            if not os.path.isfile(entry_path):
                continue
            if os.path.splitext(entry)[1].lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            img = cv2.imread(entry_path)
            crop = extract_face_crop(img, detector)
            if crop is None:
                continue

            emb = embedder.embed(crop)
            if emb is None or emb.vector is None:
                continue

            vec = emb.vector
            norm = float(np.linalg.norm(vec))
            stats.append(
                ImageEmbeddingStat(
                    identity=identity,
                    filename=entry,
                    backend=backend_name,
                    dimension=len(vec),
                    norm=norm,
                    min_val=float(np.min(vec)),
                    max_val=float(np.max(vec)),
                    mean_val=float(np.mean(vec)),
                    std_val=float(np.std(vec)),
                    vector=vec,
                )
            )

    N = len(stats)
    matrix = np.zeros((N, N), dtype=np.float32)
    same_sims: List[float] = []
    cross_sims: List[float] = []

    for i in range(N):
        for j in range(N):
            sim = float(np.dot(stats[i].vector, stats[j].vector))
            matrix[i, j] = sim
            if i < j:
                if stats[i].identity == stats[j].identity:
                    same_sims.append(sim)
                else:
                    cross_sims.append(sim)

    return stats, matrix, same_sims, cross_sims


def run_sanity_tests(ort_embedder: ONNXRuntimeArcFaceEmbedder, sample_crop: np.ndarray, diff_crop: np.ndarray) -> None:
    """Performs deterministic repeatability test, cross-identity distance test, and synthetic random image test."""
    print("\n==================================================")
    print("           ONNX RUNTIME SANITY TESTS              ")
    print("==================================================")

    # Sanity Test 1: Repeated inference determinism
    e1 = ort_embedder.embed(sample_crop)
    e2 = ort_embedder.embed(sample_crop)
    if e1 is not None and e2 is not None:
        is_deterministic = np.allclose(e1.vector, e2.vector, atol=1e-5)
        diff_max = float(np.max(np.abs(e1.vector - e2.vector)))
        print(f" 1. Determinism Test on Same Image: {'PASSED' if is_deterministic else 'FAILED'} (Max Diff: {diff_max:.2e})")
    else:
        print(" 1. Determinism Test: Skipped (no sample crop)")

    # Cross-identity distance test
    if e1 is not None and diff_crop is not None:
        e_diff = ort_embedder.embed(diff_crop)
        if e_diff is not None:
            l2_dist = float(np.linalg.norm(e1.vector - e_diff.vector))
            cos_sim = float(np.dot(e1.vector, e_diff.vector))
            print(f" 2. Distinct Identity Distance: L2 Distance = {l2_dist:.4f} | Cosine Similarity = {cos_sim:.4f}")

    # Sanity Test 2: Synthetic random RGB images
    print("\n 3. Random Synthetic Images Test (5 Distinct 112x112 Inputs):")
    np.random.seed(42)
    random_vectors = []
    for k in range(5):
        rand_img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        r_emb = ort_embedder.embed(rand_img)
        if r_emb is not None:
            random_vectors.append(r_emb.vector)

    if len(random_vectors) == 5:
        sims = []
        for i in range(5):
            for j in range(i + 1, 5):
                sims.append(float(np.dot(random_vectors[i], random_vectors[j])))
        mean_rand_sim = float(np.mean(sims))
        min_rand_sim = float(np.min(sims))
        max_rand_sim = float(np.max(sims))
        print(f"    Random Images Pairwise Cosine Sim Range: [{min_rand_sim:.4f}, {max_rand_sim:.4f}]")
        print(f"    Random Images Mean Cosine Similarity   : {mean_rand_sim:.4f}")
        if mean_rand_sim > 0.95:
            print("    [WARNING] Model produces high similarity (> 0.95) even on random noise inputs.")
        else:
            print("    [OK] Model differentiates distinct random inputs appropriately.")


def main():
    parser = argparse.ArgumentParser(
        description="VISION Slice 5.3 ArcFace Backend Comparison Diagnostic (OpenCV DNN vs ONNX Runtime)"
    )
    parser.add_argument(
        "--gallery-dir",
        type=str,
        default="data/face_gallery",
        help="Path to face gallery directory (default: data/face_gallery)",
    )
    args = parser.parse_args()

    print("==================================================")
    print(" VISION — Slice 5.3 ArcFace Backend Comparison    ")
    print("==================================================")
    print(f" Gallery Directory: {args.gallery_dir}")
    print(" Initializing OpenCV DNN and ONNX Runtime Embedders...")

    detector = FaceDetector(score_threshold=0.50)
    cv_embedder = OpenCVArcFaceEmbedder()
    ort_embedder = ONNXRuntimeArcFaceEmbedder()

    # Evaluate OpenCV DNN Backend
    cv_stats, cv_matrix, cv_same, cv_cross = evaluate_backend(
        gallery_dir=args.gallery_dir,
        embedder=cv_embedder,
        backend_name="OpenCV DNN",
        detector=detector,
    )

    # Evaluate ONNX Runtime Backend
    ort_stats, ort_matrix, ort_same, ort_cross = evaluate_backend(
        gallery_dir=args.gallery_dir,
        embedder=ort_embedder,
        backend_name="ONNX Runtime",
        detector=detector,
    )

    print("\n==================================================")
    print("           PER-IMAGE EMBEDDING STATISTICS         ")
    print("==================================================")
    print(f"{'Identity':20s} | {'Image':12s} | {'Backend':12s} | {'Dim':4s} | {'Norm':8s} | {'Min':8s} | {'Max':8s} | {'Mean':8s} | {'Std':8s}")
    print("-" * 105)

    for i in range(max(len(cv_stats), len(ort_stats))):
        if i < len(cv_stats):
            s = cv_stats[i]
            print(f"{s.identity:20s} | {s.filename:12s} | {s.backend:12s} | {s.dimension:4d} | {s.norm:8.4f} | {s.min_val:8.4f} | {s.max_val:8.4f} | {s.mean_val:8.4f} | {s.std_val:8.4f}")
        if i < len(ort_stats):
            s = ort_stats[i]
            print(f"{s.identity:20s} | {s.filename:12s} | {s.backend:12s} | {s.dimension:4d} | {s.norm:8.4f} | {s.min_val:8.4f} | {s.max_val:8.4f} | {s.mean_val:8.4f} | {s.std_val:8.4f}")
        print("-" * 105)

    def print_backend_summary(name: str, same: List[float], cross: List[float]):
        mean_g = float(np.mean(same)) if same else 0.0
        min_g = float(np.min(same)) if same else 0.0
        max_g = float(np.max(same)) if same else 0.0
        mean_i = float(np.mean(cross)) if cross else 0.0
        min_i = float(np.min(cross)) if cross else 0.0
        max_i = float(np.max(cross)) if cross else 0.0
        sep = mean_g - mean_i

        print(f"\n--- {name.upper()} ---")
        print(f" Genuine Mean    : {mean_g:.4f}")
        print(f" Genuine Min     : {min_g:.4f}")
        print(f" Genuine Max     : {max_g:.4f}")
        print(f" Impostor Mean   : {mean_i:.4f}")
        print(f" Impostor Min    : {min_i:.4f}")
        print(f" Impostor Max    : {max_i:.4f}")
        print(f" Separation Score: {sep:.4f} (Genuine Mean - Impostor Mean)")

    print("\n==================================================")
    print("      BACKEND SEPARATION METRICS COMPARISON       ")
    print("==================================================")
    print_backend_summary("OpenCV DNN", cv_same, cv_cross)
    print_backend_summary("ONNX Runtime", ort_same, ort_cross)

    # Sanity checks
    sample_crop = None
    diff_crop = None
    ath_path = os.path.join(args.gallery_dir, "Atharva_Jaysingpure", "front.jpeg")
    shr_path = os.path.join(args.gallery_dir, "Shreyas_Chavan", "front.jpeg")
    if os.path.exists(ath_path):
        sample_crop = extract_face_crop(cv2.imread(ath_path), detector)
    if os.path.exists(shr_path):
        diff_crop = extract_face_crop(cv2.imread(shr_path), detector)

    if sample_crop is None:
        sample_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)

    run_sanity_tests(ort_embedder, sample_crop, diff_crop)


if __name__ == "__main__":
    main()
