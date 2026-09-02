import argparse
import os
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple
import cv2
import numpy as np

from src.face.alignment import align_face
from src.face.detector import FaceDetector
from src.face.embedder import FaceEmbedder
from src.face.gallery import SUPPORTED_IMAGE_EXTENSIONS
from src.face.preprocessing import l2_normalize


class ModelInspectionResult(NamedTuple):
    model_path: str
    input_name: str
    input_shape: List[int]
    input_dtype: str
    output_name: str
    output_shape: List[int]
    output_dtype: str
    opset_version: str


def inspect_onnx_model(model_path: str = "models/arcface_resnet100.onnx") -> ModelInspectionResult:
    """Inspects ArcFace ONNX model metadata, input/output shapes, dtypes, and opset version."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    input_name = "data"
    input_shape = [1, 3, 112, 112]
    input_dtype = "float32"
    output_name = "fc1"
    output_shape = [1, 512]
    output_dtype = "float32"
    opset_version = "Opset 8"

    # Attempt inspecting via onnx runtime or onnx module if installed
    try:
        import onnx
        model = onnx.load(model_path)
        if model.opset_import:
            opset_version = f"Opset {model.opset_import[0].version}"
        if model.graph.input:
            inp = model.graph.input[0]
            input_name = inp.name
            dim_vals = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            if dim_vals:
                input_shape = list(dim_vals)
        if model.graph.output:
            out = model.graph.output[0]
            output_name = out.name
            dim_vals = [d.dim_value for d in out.type.tensor_type.shape.dim]
            if dim_vals:
                output_shape = list(dim_vals)
    except Exception:
        pass

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(model_path)
        inp = session.get_inputs()[0]
        out = session.get_outputs()[0]
        input_name = inp.name
        input_shape = list(inp.shape)
        input_dtype = str(inp.type)
        output_name = out.name
        output_shape = list(out.shape)
        output_dtype = str(out.type)
    except Exception:
        pass

    return ModelInspectionResult(
        model_path=model_path,
        input_name=input_name,
        input_shape=input_shape,
        input_dtype=input_dtype,
        output_name=output_name,
        output_shape=output_shape,
        output_dtype=output_dtype,
        opset_version=opset_version,
    )


def preprocess_variant_a(crop: np.ndarray) -> np.ndarray:
    """Variant A: BGR -> resize 112x112 -> float32 -> (x - 127.5) / 128.0 (NCHW BGR)"""
    resized = cv2.resize(crop, (112, 112))
    blob = (resized.astype(np.float32) - 127.5) / 128.0
    blob = np.transpose(blob, (2, 0, 1))
    return np.expand_dims(blob, axis=0)


def preprocess_variant_b(crop: np.ndarray) -> np.ndarray:
    """Variant B: BGR -> RGB -> resize 112x112 -> float32 -> (x - 127.5) / 128.0 (NCHW RGB)"""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (112, 112))
    blob = (resized.astype(np.float32) - 127.5) / 128.0
    blob = np.transpose(blob, (2, 0, 1))
    return np.expand_dims(blob, axis=0)


def preprocess_variant_c(crop: np.ndarray) -> np.ndarray:
    """Variant C: BGR -> RGB -> resize 112x112 -> float32 -> (x - 127.5) / 127.5 (NCHW RGB)"""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (112, 112))
    blob = (resized.astype(np.float32) - 127.5) / 127.5
    blob = np.transpose(blob, (2, 0, 1))
    return np.expand_dims(blob, axis=0)


def preprocess_variant_d(crop: np.ndarray) -> np.ndarray:
    """Variant D: BGR -> resize 112x112 -> float32 -> (x - 127.5) / 127.5 (NCHW BGR)"""
    resized = cv2.resize(crop, (112, 112))
    blob = (resized.astype(np.float32) - 127.5) / 127.5
    blob = np.transpose(blob, (2, 0, 1))
    return np.expand_dims(blob, axis=0)


PREPROCESSING_VARIANTS = {
    "Variant A (BGR, /128)": preprocess_variant_a,
    "Variant B (RGB, /128)": preprocess_variant_b,
    "Variant C (RGB, /127.5)": preprocess_variant_c,
    "Variant D (BGR, /127.5)": preprocess_variant_d,
}


class ConfigResult(NamedTuple):
    mode: str
    variant_name: str
    mean_genuine: float
    min_genuine: float
    max_genuine: float
    mean_impostor: float
    min_impostor: float
    max_impostor: float
    separation: float


def evaluate_configuration(
    gallery_dir: str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    mode: str,
    variant_name: str,
    preprocess_fn,
) -> ConfigResult:
    """Evaluates a specific (Mode, Preprocessing Variant) combination across the gallery."""
    identity_dirs = sorted([
        d for d in os.listdir(gallery_dir)
        if os.path.isdir(os.path.join(gallery_dir, d))
    ])

    items: List[Tuple[str, np.ndarray]] = []

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

            if mode == "ALIGNED" and best_face.landmarks is not None:
                crop = align_face(img, best_face.landmarks, (112, 112))
            else:
                fb = best_face.bbox
                h, w = img.shape[:2]
                x1, y1 = max(0, min(fb.x1, w)), max(0, min(fb.y1, h))
                x2, y2 = min(w, max(x1, fb.x2)), min(h, max(y1, fb.y2))
                crop = img[y1:y2, x1:x2]

            if crop is None or crop.size == 0:
                continue

            blob = preprocess_fn(crop)
            embedder.net.setInput(blob)
            raw = embedder.net.forward().flatten()
            vec = l2_normalize(raw)

            items.append((identity, vec))

    N = len(items)
    same_sims: List[float] = []
    cross_sims: List[float] = []

    for i in range(N):
        for j in range(i + 1, N):
            sim = float(np.dot(items[i][1], items[j][1]))
            if items[i][0] == items[j][0]:
                same_sims.append(sim)
            else:
                cross_sims.append(sim)

    mean_gen = float(np.mean(same_sims)) if same_sims else 0.0
    min_gen = float(np.min(same_sims)) if same_sims else 0.0
    max_gen = float(np.max(same_sims)) if same_sims else 0.0

    mean_imp = float(np.mean(cross_sims)) if cross_sims else 0.0
    min_imp = float(np.min(cross_sims)) if cross_sims else 0.0
    max_imp = float(np.max(cross_sims)) if cross_sims else 0.0

    separation = mean_gen - mean_imp

    return ConfigResult(
        mode=mode,
        variant_name=variant_name,
        mean_genuine=mean_gen,
        min_genuine=min_gen,
        max_genuine=max_gen,
        mean_impostor=mean_imp,
        min_impostor=min_imp,
        max_impostor=max_imp,
        separation=separation,
    )


def run_full_diagnostic(gallery_dir: str = "data/face_gallery") -> List[ConfigResult]:
    """Runs all combinations of (UNALIGNED/ALIGNED) x (Preprocessing Variants) on the gallery."""
    inspection = inspect_onnx_model()

    print("==================================================")
    print(" VISION — Slice 5.2 ArcFace Model & Alignment Audit ")
    print("==================================================")
    print(f" Model Path    : {inspection.model_path}")
    print(f" Input Tensor  : {inspection.input_name} shape={inspection.input_shape} ({inspection.input_dtype})")
    print(f" Output Tensor : {inspection.output_name} shape={inspection.output_shape} ({inspection.output_dtype})")
    print(f" ONNX Version  : {inspection.opset_version}")
    print("==================================================")

    detector = FaceDetector(score_threshold=0.50)
    embedder = FaceEmbedder()

    results: List[ConfigResult] = []

    for mode in ["UNALIGNED", "ALIGNED"]:
        for var_name, fn in PREPROCESSING_VARIANTS.items():
            res = evaluate_configuration(
                gallery_dir=gallery_dir,
                detector=detector,
                embedder=embedder,
                mode=mode,
                variant_name=var_name,
                preprocess_fn=fn,
            )
            results.append(res)

    print("\n==================================================")
    print("   PREPROCESSING & ALIGNMENT COMPARISON MATRIX    ")
    print("==================================================")
    print(f"{'Mode':10s} | {'Variant':23s} | {'Gen Mean':8s} | {'Imp Mean':8s} | {'Separation':10s}")
    print("-" * 72)
    for r in results:
        print(f"{r.mode:10s} | {r.variant_name:23s} | {r.mean_genuine:8.4f} | {r.mean_impostor:8.4f} | {r.separation:10.4f}")

    sorted_res = sorted(results, key=lambda x: x.separation, reverse=True)
    best = sorted_res[0]

    print("\n==================================================")
    print("                BEST CONFIGURATION                ")
    print("==================================================")
    print(f" Best Mode        : {best.mode}")
    print(f" Best Variant     : {best.variant_name}")
    print(f" Mean Genuine Sim : {best.mean_genuine:.4f} (Range: [{best.min_genuine:.4f}, {best.max_genuine:.4f}])")
    print(f" Mean Impostor Sim: {best.mean_impostor:.4f} (Range: [{best.min_impostor:.4f}, {best.max_impostor:.4f}])")
    print(f" Separation Score : {best.separation:.4f}")
    print("==================================================")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="VISION Slice 5.2 ArcFace Input Validation and Alignment Diagnostic"
    )
    parser.add_argument(
        "--gallery-dir",
        type=str,
        default="data/face_gallery",
        help="Path to face gallery directory (default: data/face_gallery)",
    )
    args = parser.parse_args()

    run_full_diagnostic(gallery_dir=args.gallery_dir)


if __name__ == "__main__":
    main()
