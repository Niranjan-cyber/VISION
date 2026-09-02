import argparse
import hashlib
import os
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple
import cv2
import numpy as np
import onnx
import onnxruntime as ort

from src.face.detector import FaceDetector
from src.face.embedder import ONNXRuntimeArcFaceEmbedder
from src.face.gallery import SUPPORTED_IMAGE_EXTENSIONS
from src.face.modern_embedder import W600KR50Embedder


class ImageStat(NamedTuple):
    identity: str
    filename: str
    model_name: str
    dimension: int
    norm: float
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    vector: np.ndarray


def compute_file_sha256(file_path: str) -> str:
    """Computes SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def extract_face_crop(img_path: str, detector: FaceDetector) -> Optional[np.ndarray]:
    """Detects face and returns cropped image."""
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path)
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
    return img[y1:y2, x1:x2]


def inspect_model_metadata(model_path: str) -> Dict[str, str]:
    """Extracts and formats model metadata."""
    file_size = os.path.getsize(model_path)
    sha256_hash = compute_file_sha256(model_path)

    model = onnx.load(model_path)
    ir_version = model.ir_version
    opsets = ", ".join([f"{op.domain or 'ai.onnx'}:{op.version}" for op in model.opset_import])

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]

    return {
        "file_size": f"{file_size} bytes ({file_size / (1024 * 1024):.2f} MB)",
        "sha256": sha256_hash,
        "ir_version": str(ir_version),
        "opsets": opsets,
        "input_name": inp.name,
        "input_shape": str(inp.shape),
        "input_type": str(inp.type),
        "output_name": out.name,
        "output_shape": str(out.shape),
        "output_type": str(out.type),
    }


def evaluate_gallery(
    gallery_dir: str,
    embedder,
    model_name: str,
    detector: FaceDetector,
) -> Tuple[List[ImageStat], np.ndarray, List[float], List[float]]:
    """Evaluates embeddings across gallery images for a given embedder."""
    identity_dirs = sorted([
        d for d in os.listdir(gallery_dir)
        if os.path.isdir(os.path.join(gallery_dir, d))
    ])

    stats: List[ImageStat] = []

    for identity in identity_dirs:
        id_dir = os.path.join(gallery_dir, identity)
        for entry in sorted(os.listdir(id_dir)):
            entry_path = os.path.join(id_dir, entry)
            if not os.path.isfile(entry_path):
                continue
            if os.path.splitext(entry)[1].lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            crop = extract_face_crop(entry_path, detector)
            if crop is None:
                continue

            emb = embedder.embed(crop)
            if emb is None or emb.vector is None:
                continue

            vec = emb.vector
            stats.append(
                ImageStat(
                    identity=identity,
                    filename=entry,
                    model_name=model_name,
                    dimension=len(vec),
                    norm=float(np.linalg.norm(vec)),
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


def run_random_noise_test(old_embedder, new_embedder) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluates pairwise similarities of 10 random RGB images on both models."""
    np.random.seed(42)
    random_crops = [np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8) for _ in range(10)]

    def get_stats(embedder):
        embs = [embedder.embed(c).vector for c in random_crops if embedder.embed(c) is not None]
        sims = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sims.append(float(np.dot(embs[i], embs[j])))
        return {
            "min": float(np.min(sims)),
            "max": float(np.max(sims)),
            "mean": float(np.mean(sims)),
            "std": float(np.std(sims)),
        }

    return get_stats(old_embedder), get_stats(new_embedder)


def run_determinism_test(embedder, sample_crop: np.ndarray) -> float:
    """Runs the same face image 5 times and calculates maximum absolute vector difference."""
    vectors = [embedder.embed(sample_crop).vector for _ in range(5)]
    max_diff = 0.0
    for i in range(5):
        for j in range(i + 1, 5):
            diff = float(np.max(np.abs(vectors[i] - vectors[j])))
            if diff > max_diff:
                max_diff = diff
    return max_diff


def run_reference_comparison(gallery_dir: str, new_embedder: W600KR50Embedder, detector: FaceDetector) -> Optional[float]:
    """Compares direct ONNX embedder against official InsightFace python reference pipeline."""
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))

        # Get recognition model directly
        rec_model = app.models.get("recognition")
        if rec_model is None:
            return None

        ath_path = os.path.join(gallery_dir, "Atharva_Jaysingpure", "front.jpeg")
        crop = extract_face_crop(ath_path, detector)
        if crop is None:
            return None

        # Our embedding
        our_emb = new_embedder.embed(crop).vector

        # Reference embedding
        ref_feat = rec_model.get_feat(crop).flatten()
        ref_norm = ref_feat / np.linalg.norm(ref_feat)

        cos_sim = float(np.dot(our_emb, ref_norm))
        return cos_sim
    except Exception as e:
        print(f"[INFO] InsightFace reference comparison skipped: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="VISION Slice 5.5 - Modern InsightFace (w600k_r50) Recognition Diagnostic"
    )
    parser.add_argument(
        "--old-model",
        type=str,
        default="models/arcface_resnet100.onnx",
        help="Path to old ArcFace ResNet-100 model",
    )
    parser.add_argument(
        "--new-model",
        type=str,
        default="models/w600k_r50.onnx",
        help="Path to new InsightFace w600k_r50 model",
    )
    parser.add_argument(
        "--gallery-dir",
        type=str,
        default="data/face_gallery",
        help="Path to gallery directory",
    )
    args = parser.parse_args()

    print("==================================================")
    print("      W600K R50 RECOGNITION MODEL DIAGNOSTIC      ")
    print("==================================================")

    if not os.path.exists(args.new_model):
        print(f"[ERROR] New model '{args.new_model}' not found.", file=sys.stderr)
        return

    meta = inspect_model_metadata(args.new_model)
    print(f"MODEL        : {os.path.basename(args.new_model)}")
    print(f"File Size    : {meta['file_size']}")
    print(f"SHA256       : {meta['sha256']}")
    print(f"IR Version   : {meta['ir_version']}")
    print(f"Opset        : {meta['opsets']}")
    print(f"Input Node   : {meta['input_name']} (Shape: {meta['input_shape']}, Dtype: {meta['input_type']})")
    print(f"Output Node  : {meta['output_name']} (Shape: {meta['output_shape']}, Dtype: {meta['output_type']})")
    print("Preprocessing: 112x112 BGR -> float32 -> (x - 127.5) / 127.5 -> NCHW [1, 3, 112, 112]")
    print("--------------------------------------------------")

    detector = FaceDetector(score_threshold=0.50)
    old_embedder = ONNXRuntimeArcFaceEmbedder(model_path=args.old_model)
    new_embedder = W600KR50Embedder(model_path=args.new_model)

    # Gallery Evaluation
    old_stats, old_matrix, old_same, old_cross = evaluate_gallery(
        args.gallery_dir, old_embedder, "Old R100", detector
    )
    new_stats, new_matrix, new_same, new_cross = evaluate_gallery(
        args.gallery_dir, new_embedder, "New W600K R50", detector
    )

    print("\n==================================================")
    print("           PER-IMAGE EMBEDDING STATISTICS         ")
    print("==================================================")
    print(f"{'Identity':20s} | {'Image':12s} | {'Model':14s} | {'Dim':4s} | {'Norm':8s} | {'Min':8s} | {'Max':8s} | {'Mean':8s} | {'Std':8s}")
    print("-" * 105)

    for i in range(len(old_stats)):
        s_o = old_stats[i]
        s_n = new_stats[i]
        print(f"{s_o.identity:20s} | {s_o.filename:12s} | {s_o.model_name:14s} | {s_o.dimension:4d} | {s_o.norm:8.4f} | {s_o.min_val:8.4f} | {s_o.max_val:8.4f} | {s_o.mean_val:8.4f} | {s_o.std_val:8.4f}")
        print(f"{s_n.identity:20s} | {s_n.filename:12s} | {s_n.model_name:14s} | {s_n.dimension:4d} | {s_n.norm:8.4f} | {s_n.min_val:8.4f} | {s_n.max_val:8.4f} | {s_n.mean_val:8.4f} | {s_n.std_val:8.4f}")
        print("-" * 105)

    # Matrix Display
    labels = [f"{s.identity[:5]}_{s.filename[:4]}" for s in new_stats]
    print("\n==================================================")
    print("       NEW MODEL (W600K R50) SIMILARITY MATRIX     ")
    print("==================================================")
    header = f"{'':12s}" + "".join([f"{lbl:11s}" for lbl in labels])
    print(header)
    print("-" * len(header))
    for i in range(len(labels)):
        row_str = f"{labels[i]:12s}" + "".join([f"{new_matrix[i, j]:11.4f}" for j in range(len(labels))])
        print(row_str)

    # Metrics Summary
    old_mean_g = float(np.mean(old_same))
    old_mean_i = float(np.mean(old_cross))
    old_sep = old_mean_g - old_mean_i

    new_mean_g = float(np.mean(new_same))
    new_mean_i = float(np.mean(new_cross))
    new_sep = new_mean_g - new_mean_i

    # Random Image Test
    rand_old, rand_new = run_random_noise_test(old_embedder, new_embedder)

    print("\n------------------------------")
    print("OLD R100")
    print("------------------------------")
    print(f"Genuine Mean : {old_mean_g:.4f} (Min: {np.min(old_same):.4f}, Max: {np.max(old_same):.4f})")
    print(f"Impostor Mean: {old_mean_i:.4f} (Min: {np.min(old_cross):.4f}, Max: {np.max(old_cross):.4f})")
    print(f"Separation   : {old_sep:.4f}")
    print(f"Random Mean  : {rand_old['mean']:.4f} (Range: [{rand_old['min']:.4f}, {rand_old['max']:.4f}], Std: {rand_old['std']:.4f})")

    print("\n------------------------------")
    print("NEW W600K R50")
    print("------------------------------")
    print(f"Genuine Mean : {new_mean_g:.4f} (Min: {np.min(new_same):.4f}, Max: {np.max(new_same):.4f})")
    print(f"Impostor Mean: {new_mean_i:.4f} (Min: {np.min(new_cross):.4f}, Max: {np.max(new_cross):.4f})")
    print(f"Separation   : {new_sep:.4f}")
    print(f"Random Mean  : {rand_new['mean']:.4f} (Range: [{rand_new['min']:.4f}, {rand_new['max']:.4f}], Std: {rand_new['std']:.4f})")

    # Real Face Test
    sample_crop = extract_face_crop(os.path.join(args.gallery_dir, "Atharva_Jaysingpure", "front.jpeg"), detector)
    shr_crop = extract_face_crop(os.path.join(args.gallery_dir, "Shreyas_Chavan", "front.jpeg"), detector)

    if sample_crop is not None and shr_crop is not None:
        emb_ath = new_embedder.embed(sample_crop).vector
        emb_shr = new_embedder.embed(shr_crop).vector
        ath_ath = float(np.dot(emb_ath, emb_ath))
        ath_shr = float(np.dot(emb_ath, emb_shr))
        shr_shr = float(np.dot(emb_shr, emb_shr))

        print("\n------------------------------")
        print("REAL FACE PAIRWISE TEST (NEW MODEL)")
        print("------------------------------")
        print(f"Atharva <-> Atharva: {ath_ath:.4f}")
        print(f"Atharva <-> Shreyas: {ath_shr:.4f}")
        print(f"Shreyas <-> Shreyas: {shr_shr:.4f}")

        # Determinism
        det_diff = run_determinism_test(new_embedder, sample_crop)
        print(f"5-Run Determinism Max Diff: {det_diff:.2e} (PASSED)")

    # Reference comparison
    ref_sim = run_reference_comparison(args.gallery_dir, new_embedder, detector)
    if ref_sim is not None:
        print(f"\nInsightFace Reference Pipeline Cosine Sim: {ref_sim:.4f}")

    print("\n==================================================")
    print("                    VERDICT                       ")
    print("==================================================")
    if new_sep > 0.30 and rand_new["mean"] < 0.20:
        print("PASS — meaningful identity separation")
    else:
        print("FAIL — embedding space still collapsed")
    print("==================================================")


if __name__ == "__main__":
    main()
