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
from src.face.gallery import SUPPORTED_IMAGE_EXTENSIONS

OFFICIAL_SHA256 = "f3a6bc281e72f88862f5748b53be3d76b3b48f8f1ab1f4a537941bdc4e1b01da"
OFFICIAL_FILE_SIZE = 261036388


def compute_file_sha256(file_path: str) -> str:
    """Computes SHA-256 hash of a file in 64KB blocks."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


class ModelIntegrityInfo(NamedTuple):
    file_path: str
    file_size: int
    sha256_hash: str
    hash_match: bool
    ir_version: int
    producer_name: str
    producer_version: str
    opsets: List[str]
    total_nodes: int
    bn_nodes_count: int
    bn_spatial_0_count: int
    bn_spatial_1_count: int
    bn_no_spatial_attr_count: int


def inspect_model_integrity(model_path: str = "models/arcface_resnet100.onnx") -> ModelIntegrityInfo:
    """Inspects ONNX model integrity, metadata, nodes, and BatchNormalization spatial attributes."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    file_size = os.path.getsize(model_path)
    sha256_hash = compute_file_sha256(model_path)
    hash_match = (sha256_hash.lower() == OFFICIAL_SHA256.lower())

    model = onnx.load(model_path)
    ir_version = model.ir_version
    producer_name = model.producer_name or "Unknown"
    producer_version = model.producer_version or "Unknown"
    opsets = [f"domain='{op.domain}' version={op.version}" for op in model.opset_import]

    total_nodes = len(model.graph.node)
    bn_nodes_count = 0
    bn_spatial_0_count = 0
    bn_spatial_1_count = 0
    bn_no_spatial_attr_count = 0

    for node in model.graph.node:
        if node.op_type == "BatchNormalization":
            bn_nodes_count += 1
            spatial_attr = None
            for attr in node.attribute:
                if attr.name == "spatial":
                    spatial_attr = attr.i
                    break
            if spatial_attr == 0:
                bn_spatial_0_count += 1
            elif spatial_attr == 1:
                bn_spatial_1_count += 1
            else:
                bn_no_spatial_attr_count += 1

    return ModelIntegrityInfo(
        file_path=model_path,
        file_size=file_size,
        sha256_hash=sha256_hash,
        hash_match=hash_match,
        ir_version=ir_version,
        producer_name=producer_name,
        producer_version=producer_version,
        opsets=opsets,
        total_nodes=total_nodes,
        bn_nodes_count=bn_nodes_count,
        bn_spatial_0_count=bn_spatial_0_count,
        bn_spatial_1_count=bn_spatial_1_count,
        bn_no_spatial_attr_count=bn_no_spatial_attr_count,
    )


def print_bn_node_details(model_path: str, max_nodes: int = 5) -> None:
    """Prints detailed attributes for the first several BatchNormalization nodes."""
    model = onnx.load(model_path)
    count = 0
    print(f"\n--- First {max_nodes} BatchNormalization Nodes Details ---")
    for node in model.graph.node:
        if node.op_type == "BatchNormalization":
            count += 1
            spatial = "Not Set"
            epsilon = "Default"
            momentum = "Default"
            for attr in node.attribute:
                if attr.name == "spatial":
                    spatial = str(attr.i)
                elif attr.name == "epsilon":
                    epsilon = str(attr.f)
                elif attr.name == "momentum":
                    momentum = str(attr.f)
            print(f" Node [{count}]: '{node.name or node.output[0]}' | spatial={spatial} | epsilon={epsilon} | momentum={momentum} | inputs={len(node.input)}")
            if count >= max_nodes:
                break


def print_graph_tail_nodes(model_path: str, tail_count: int = 10) -> None:
    """Prints the final graph nodes leading to output fc1."""
    model = onnx.load(model_path)
    nodes = model.graph.node
    start_idx = max(0, len(nodes) - tail_count)
    print(f"\n--- Final {tail_count} Graph Nodes Leading to Output ---")
    for idx in range(start_idx, len(nodes)):
        node = nodes[idx]
        print(f" Node #{idx:3d}: op_type={node.op_type:20s} name='{node.name}' inputs={list(node.input)} outputs={list(node.output)}")


def create_patched_model(
    original_path: str = "models/arcface_resnet100.onnx",
    patched_path: str = "models/arcface_resnet100_patched.onnx",
) -> Tuple[int, int]:
    """
    Creates a patched copy of the ArcFace ONNX model where BatchNormalization nodes
    with spatial == 0 are set to spatial == 1.
    Never overwrites the original model.
    """
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original model not found at '{original_path}'")

    model = onnx.load(original_path)
    changed_count = 0
    total_bn_count = 0

    for node in model.graph.node:
        if node.op_type == "BatchNormalization":
            total_bn_count += 1
            for attr in node.attribute:
                if attr.name == "spatial" and attr.i == 0:
                    attr.i = 1
                    changed_count += 1

    onnx.checker.check_model(model)
    onnx.save(model, patched_path)
    return changed_count, total_bn_count


def extract_face_crop(img_path: str, detector: FaceDetector) -> Optional[np.ndarray]:
    """Helper to load image and extract face crop."""
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


def run_session_inference(
    session: ort.InferenceSession,
    input_tensor: np.ndarray,
) -> np.ndarray:
    """Runs inference and returns 512-D L2-normalized embedding."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_tensor})
    raw = outputs[0].flatten().astype(np.float32)
    norm = float(np.linalg.norm(raw))
    if norm > 1e-12:
        return raw / norm
    return np.zeros_like(raw)


def build_tensor(crop: np.ndarray, is_rgb: bool = True, normalize: bool = True) -> np.ndarray:
    """Builds (1, 3, 112, 112) float32 tensor from face crop."""
    resized = cv2.resize(crop, (112, 112))
    if is_rgb:
        img_fmt = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    else:
        img_fmt = resized

    f_img = img_fmt.astype(np.float32)
    if normalize:
        tensor_data = (f_img - 127.5) / 128.0
    else:
        tensor_data = f_img

    chw = np.transpose(tensor_data, (2, 0, 1))
    return np.ascontiguousarray(np.expand_dims(chw, axis=0), dtype=np.float32)


def run_comparative_experiment(
    orig_path: str = "models/arcface_resnet100.onnx",
    patched_path: str = "models/arcface_resnet100_patched.onnx",
    gallery_dir: str = "data/face_gallery",
) -> None:
    """Runs comparative experiments on original vs patched model across face images and random inputs."""
    print("\n==================================================")
    print("   ORIGINAL VS PATCHED ARCFACE MODEL COMPARISON   ")
    print("==================================================")

    detector = FaceDetector(score_threshold=0.50)
    ath_path = os.path.join(gallery_dir, "Atharva_Jaysingpure", "front.jpeg")
    shr_path = os.path.join(gallery_dir, "Shreyas_Chavan", "front.jpeg")

    crop_ath = extract_face_crop(ath_path, detector)
    crop_shr = extract_face_crop(shr_path, detector)

    if crop_ath is None:
        crop_ath = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    if crop_shr is None:
        crop_shr = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)

    session_orig = ort.InferenceSession(orig_path, providers=["CPUExecutionProvider"])
    session_patch = ort.InferenceSession(patched_path, providers=["CPUExecutionProvider"])

    tensor_ath = build_tensor(crop_ath, is_rgb=True, normalize=True)
    tensor_shr = build_tensor(crop_shr, is_rgb=True, normalize=True)

    # Atharva vs Shreyas
    emb_ath_orig = run_session_inference(session_orig, tensor_ath)
    emb_shr_orig = run_session_inference(session_orig, tensor_shr)
    sim_orig = float(np.dot(emb_ath_orig, emb_shr_orig))

    emb_ath_patch = run_session_inference(session_patch, tensor_ath)
    emb_shr_patch = run_session_inference(session_patch, tensor_shr)
    sim_patch = float(np.dot(emb_ath_patch, emb_shr_patch))

    print(f" Atharva <-> Shreyas Cosine Similarity:")
    print(f"   Original Model : {sim_orig:.4f}")
    print(f"   Patched Model  : {sim_patch:.4f}")

    # Random noise images test
    np.random.seed(42)
    rand_tensors = [build_tensor(np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)) for _ in range(5)]

    def calc_rand_sims(session):
        embs = [run_session_inference(session, t) for t in rand_tensors]
        sims = []
        for i in range(5):
            for j in range(i + 1, 5):
                sims.append(float(np.dot(embs[i], embs[j])))
        return float(np.min(sims)), float(np.max(sims)), float(np.mean(sims))

    min_o, max_o, mean_o = calc_rand_sims(session_orig)
    min_p, max_p, mean_p = calc_rand_sims(session_patch)

    print(f"\n Random Synthetic Images Pairwise Similarity:")
    print(f"   Original Model : Range = [{min_o:.4f}, {max_o:.4f}] | Mean = {mean_o:.4f}")
    print(f"   Patched Model  : Range = [{min_p:.4f}, {max_p:.4f}] | Mean = {mean_p:.4f}")

    # Controlled Unnormalized Input Experiment
    print("\n--- Controlled Preprocessing Variants on Both Models ---")
    variants = {
        "A: RGB raw [0, 255]": (True, False),
        "B: RGB (x-127.5)/128": (True, True),
        "C: BGR raw [0, 255]": (False, False),
        "D: BGR (x-127.5)/128": (False, True),
    }

    for name, (is_rgb, norm) in variants.items():
        t_a = build_tensor(crop_ath, is_rgb, norm)
        t_s = build_tensor(crop_shr, is_rgb, norm)
        s_o = float(np.dot(run_session_inference(session_orig, t_a), run_session_inference(session_orig, t_s)))
        s_p = float(np.dot(run_session_inference(session_patch, t_a), run_session_inference(session_patch, t_s)))
        print(f"   {name:25s} | Orig Sim: {s_o:.4f} | Patched Sim: {s_p:.4f}")

    print("\n==================================================")
    print("                    VERDICT                       ")
    print("==================================================")
    if sim_patch < 0.85 and mean_p < 0.50:
        print(" VERDICT: MODEL VALID (Patch successfully restored identity discrimination!)")
    else:
        print(" VERDICT: MODEL STILL BROKEN (BatchNorm spatial patch alone does not resolve vector collapse)")
    print("==================================================")


def main():
    parser = argparse.ArgumentParser(
        description="VISION Slice 5.4 ArcFace Model Integrity & Repair Diagnostic"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/arcface_resnet100.onnx",
        help="Path to original ArcFace ONNX model (default: models/arcface_resnet100.onnx)",
    )
    parser.add_argument(
        "--patched-path",
        type=str,
        default="models/arcface_resnet100_patched.onnx",
        help="Path to save patched ArcFace ONNX model (default: models/arcface_resnet100_patched.onnx)",
    )
    parser.add_argument(
        "--gallery-dir",
        type=str,
        default="data/face_gallery",
        help="Path to face gallery directory (default: data/face_gallery)",
    )
    args = parser.parse_args()

    print("==================================================")
    print("        ARCFACE MODEL INTEGRITY DIAGNOSTIC        ")
    print("==================================================")
    info = inspect_model_integrity(args.model_path)
    print(f" Model Path          : {info.file_path}")
    print(f" File Size           : {info.file_size} bytes (Expected: {OFFICIAL_FILE_SIZE})")
    print(f" SHA256 Hash         : {info.sha256_hash}")
    print(f" Official SHA256     : {OFFICIAL_SHA256}")
    print(f" Hash Match          : {'YES' if info.hash_match else 'NO (MODEL HASH MISMATCH)'}")
    print(f" ONNX IR Version     : {info.ir_version}")
    print(f" Producer            : {info.producer_name} ({info.producer_version})")
    print(f" Opset Imports       : {info.opsets}")
    print(f" Total Graph Nodes   : {info.total_nodes}")
    print(f" BatchNorm Nodes     : {info.bn_nodes_count}")
    print(f"   spatial == 0      : {info.bn_spatial_0_count}")
    print(f"   spatial == 1      : {info.bn_spatial_1_count}")
    print(f"   spatial unassigned: {info.bn_no_spatial_attr_count}")

    print_bn_node_details(args.model_path, max_nodes=5)
    print_graph_tail_nodes(args.model_path, tail_count=10)

    if not info.hash_match:
        print("\n[ERROR] MODEL HASH MISMATCH. Aborting patch experiment.", file=sys.stderr)
        return

    # Task 3: Create Patched Copy
    print("\n[INFO] Creating patched model copy (spatial=0 -> spatial=1)...")
    changed_count, total_bn = create_patched_model(args.model_path, args.patched_path)
    print(f"[INFO] Successfully patched {changed_count}/{total_bn} BatchNormalization nodes.")
    print(f"[INFO] Patched model validated and saved to '{args.patched_path}'.")

    # Check patched model integrity info
    info_patched = inspect_model_integrity(args.patched_path)
    print(f"\n Patched Model BatchNorm Breakdown:")
    print(f"   spatial == 0: {info_patched.bn_spatial_0_count}")
    print(f"   spatial == 1: {info_patched.bn_spatial_1_count}")

    # Task 4 & 5: Run comparative experiment
    run_comparative_experiment(args.model_path, args.patched_path, args.gallery_dir)


if __name__ == "__main__":
    main()
