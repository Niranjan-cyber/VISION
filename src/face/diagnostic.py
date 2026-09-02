import argparse
import os
import sys
from typing import Dict, List, NamedTuple, Tuple
import cv2
import numpy as np

from src.face.detector import FaceDetector
from src.face.embedder import FaceEmbedder
from src.face.gallery import SUPPORTED_IMAGE_EXTENSIONS


class ReferenceEmbedding(NamedTuple):
    identity: str
    filename: str
    vector: np.ndarray


class DiagnosticReport(NamedTuple):
    num_identities: int
    num_embeddings: int
    embeddings: List[ReferenceEmbedding]
    similarity_matrix: np.ndarray
    same_identity_sims: List[float]
    cross_identity_sims: List[float]
    min_genuine: float
    max_genuine: float
    mean_genuine: float
    std_genuine: float
    min_impostor: float
    max_impostor: float
    mean_impostor: float
    std_impostor: float
    separation: float


def compute_pairwise_diagnostic(
    gallery_dir: str = "data/face_gallery",
    detector: FaceDetector = None,
    embedder: FaceEmbedder = None,
) -> DiagnosticReport:
    """
    Scans reference images in gallery_dir, extracts 512-D ArcFace embeddings,
    computes pairwise cosine similarities, and separates genuine vs impostor metrics.
    """
    if detector is None:
        detector = FaceDetector(score_threshold=0.50)

    if embedder is None:
        embedder = FaceEmbedder()

    enrolled: List[ReferenceEmbedding] = []

    if not os.path.exists(gallery_dir) or not os.path.isdir(gallery_dir):
        print(f"[WARNING] Gallery directory '{gallery_dir}' does not exist.", file=sys.stderr)
        return DiagnosticReport(
            num_identities=0,
            num_embeddings=0,
            embeddings=[],
            similarity_matrix=np.zeros((0, 0), dtype=np.float32),
            same_identity_sims=[],
            cross_identity_sims=[],
            min_genuine=0.0,
            max_genuine=0.0,
            mean_genuine=0.0,
            std_genuine=0.0,
            min_impostor=0.0,
            max_impostor=0.0,
            mean_impostor=0.0,
            std_impostor=0.0,
            separation=0.0,
        )

    identity_dirs = [
        d for d in os.listdir(gallery_dir)
        if os.path.isdir(os.path.join(gallery_dir, d))
    ]

    for identity in sorted(identity_dirs):
        id_dir = os.path.join(gallery_dir, identity)
        entries = sorted(os.listdir(id_dir))
        for entry in entries:
            entry_path = os.path.join(id_dir, entry)
            if not os.path.isfile(entry_path):
                continue
            ext = os.path.splitext(entry)[1].lower()
            if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            img = cv2.imread(entry_path)
            if img is None or img.size == 0:
                continue

            faces = detector.detect(img)
            if not faces:
                continue

            best_face = max(faces, key=lambda f: f.confidence)
            fb = best_face.bbox
            h, w = img.shape[:2]
            fx1, fy1 = max(0, min(fb.x1, w)), max(0, min(fb.y1, h))
            fx2, fy2 = max(0, min(fb.x2, w)), max(0, min(fb.y2, h))

            if fx2 <= fx1 or fy2 <= fy1:
                continue

            face_crop = img[fy1:fy2, fx1:fx2]
            if face_crop.size == 0:
                continue

            emb = embedder.embed(face_crop)
            if emb is not None and emb.vector is not None:
                enrolled.append(
                    ReferenceEmbedding(
                        identity=identity,
                        filename=entry,
                        vector=np.copy(emb.vector),
                    )
                )

    N = len(enrolled)
    matrix = np.zeros((N, N), dtype=np.float32)

    same_identity_sims: List[float] = []
    cross_identity_sims: List[float] = []

    for i in range(N):
        for j in range(N):
            sim = float(np.dot(enrolled[i].vector, enrolled[j].vector))
            matrix[i, j] = sim

            if i < j:
                if enrolled[i].identity == enrolled[j].identity:
                    same_identity_sims.append(sim)
                else:
                    cross_identity_sims.append(sim)

    num_identities = len(set(e.identity for e in enrolled))

    min_gen = float(np.min(same_identity_sims)) if same_identity_sims else 0.0
    max_gen = float(np.max(same_identity_sims)) if same_identity_sims else 0.0
    mean_gen = float(np.mean(same_identity_sims)) if same_identity_sims else 0.0
    std_gen = float(np.std(same_identity_sims)) if same_identity_sims else 0.0

    min_imp = float(np.min(cross_identity_sims)) if cross_identity_sims else 0.0
    max_imp = float(np.max(cross_identity_sims)) if cross_identity_sims else 0.0
    mean_imp = float(np.mean(cross_identity_sims)) if cross_identity_sims else 0.0
    std_imp = float(np.std(cross_identity_sims)) if cross_identity_sims else 0.0

    separation = mean_gen - mean_imp

    return DiagnosticReport(
        num_identities=num_identities,
        num_embeddings=N,
        embeddings=enrolled,
        similarity_matrix=matrix,
        same_identity_sims=same_identity_sims,
        cross_identity_sims=cross_identity_sims,
        min_genuine=min_gen,
        max_genuine=max_gen,
        mean_genuine=mean_gen,
        std_genuine=std_gen,
        min_impostor=min_imp,
        max_impostor=max_imp,
        mean_impostor=mean_imp,
        std_impostor=std_imp,
        separation=separation,
    )


def print_diagnostic_report(report: DiagnosticReport) -> None:
    """Prints formatted diagnostic matrix and genuine vs impostor summary."""
    print("==================================================")
    print("      ArcFace 512-D Embedding Diagnostic          ")
    print("==================================================")
    print(f" Total Identities Enrolled: {report.num_identities}")
    print(f" Total Valid Embeddings   : {report.num_embeddings}")
    print("==================================================")

    if report.num_embeddings == 0:
        print("[INFO] No valid embeddings found in gallery.")
        return

    print("\n--- ENROLLED EMBEDDINGS LIST ---")
    for idx, item in enumerate(report.embeddings):
        v_norm = np.linalg.norm(item.vector)
        print(f" [{idx:2d}] {item.identity:22s} | {item.filename:25s} | Dim: {len(item.vector)} | Norm: {v_norm:.6f}")

    print("\n--- PAIRWISE COSINE SIMILARITY MATRIX ---")
    header = "     " + "".join([f"[{i:2d}]  " for i in range(report.num_embeddings)])
    print(header)
    for i in range(report.num_embeddings):
        row_str = f"[{i:2d}] "
        for j in range(report.num_embeddings):
            sim = report.similarity_matrix[i, j]
            row_str += f"{sim:.4f} "
        print(row_str)

    print("\n--- GENUINE VS IMPOSTOR SEPARATION METRICS ---")
    print(f" Same-Identity Pairs (Genuine)  : Count = {len(report.same_identity_sims)}")
    if report.same_identity_sims:
        print(f"   Min Genuine Similarity       : {report.min_genuine:.4f}")
        print(f"   Max Genuine Similarity       : {report.max_genuine:.4f}")
        print(f"   Mean Genuine Similarity      : {report.mean_genuine:.4f} (+/- {report.std_genuine:.4f})")
    else:
        print("   No same-identity pairs found (e.g. only 1 image per identity).")

    print(f" Cross-Identity Pairs (Impostor): Count = {len(report.cross_identity_sims)}")
    if report.cross_identity_sims:
        print(f"   Min Impostor Similarity      : {report.min_impostor:.4f}")
        print(f"   Max Impostor Similarity      : {report.max_impostor:.4f}")
        print(f"   Mean Impostor Similarity     : {report.mean_impostor:.4f} (+/- {report.std_impostor:.4f})")
    else:
        print("   No cross-identity pairs found (e.g. only 1 identity enrolled).")

    print(f" Genuine/Impostor Separation    : {report.separation:.4f} (mean_genuine - mean_impostor)")
    print("==================================================")


def main():
    parser = argparse.ArgumentParser(
        description="VISION ArcFace Face Embedding Diagnostic Utility"
    )
    parser.add_argument(
        "--gallery-dir",
        type=str,
        default="data/face_gallery",
        help="Path to face gallery directory (default: data/face_gallery)",
    )
    args = parser.parse_args()

    report = compute_pairwise_diagnostic(gallery_dir=args.gallery_dir)
    print_diagnostic_report(report)


if __name__ == "__main__":
    main()
