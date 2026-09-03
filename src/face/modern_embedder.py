import os
import sys
import tempfile
import urllib.request
import zipfile
from typing import Optional
import numpy as np
import onnxruntime as ort

from src.core.types import FaceEmbedding
from src.face.w600k_preprocessing import preprocess_w600k_crop

# Official InsightFace "buffalo_l" model pack, which bundles w600k_r50.onnx.
BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
BUFFALO_L_W600K_MEMBER = "w600k_r50.onnx"


def _download_w600k_r50(target_path: str) -> None:
    """
    Downloads the official InsightFace buffalo_l model pack and extracts
    w600k_r50.onnx to target_path. Mirrors the auto-download fallback already
    used for the YuNet face detector in src/face/detector.py.
    """
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    print(
        f"[INFO] Face recognition model not found locally at '{target_path}'. "
        f"Downloading InsightFace buffalo_l pack (this is a one-time ~275MB download)...",
        file=sys.stderr,
    )
    tmp_fd, tmp_zip_path = tempfile.mkstemp(suffix=".zip")
    os.close(tmp_fd)
    try:
        urllib.request.urlretrieve(BUFFALO_L_URL, tmp_zip_path)
        with zipfile.ZipFile(tmp_zip_path) as zf:
            with zf.open(BUFFALO_L_W600K_MEMBER) as src, open(target_path, "wb") as dst:
                dst.write(src.read())
        print(f"[INFO] Successfully extracted '{BUFFALO_L_W600K_MEMBER}' to '{target_path}'.", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Failed to download/extract face recognition model: {e}", file=sys.stderr)
        raise RuntimeError(
            f"Unable to obtain '{BUFFALO_L_W600K_MEMBER}' from {BUFFALO_L_URL}"
        ) from e
    finally:
        if os.path.exists(tmp_zip_path):
            os.remove(tmp_zip_path)


class W600KR50Embedder:
    """
    Modern InsightFace ArcFace ResNet-50 (w600k_r50.onnx) feature extractor using ONNX Runtime.
    Extracts 512-dimensional L2-normalized identity embeddings.
    """

    def __init__(self, model_path: str = "models/w600k_r50.onnx") -> None:
        self.model_path = model_path

        if not os.path.exists(self.model_path):
            _download_w600k_r50(self.model_path)

        # Initialize ONNX Runtime Inference Session
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 2

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        self.input_name = inputs[0].name
        self.input_shape = inputs[0].shape
        self.input_type = inputs[0].type

        self.output_name = outputs[0].name
        self.output_shape = outputs[0].shape
        self.output_type = outputs[0].type

        # Typically 512 dimensions for w600k_r50
        self.embedding_dim = (
            self.output_shape[-1]
            if (self.output_shape and len(self.output_shape) > 1 and isinstance(self.output_shape[-1], int))
            else 512
        )

    def preprocess(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """Preprocesses face crop into [1, 3, 112, 112] float32 tensor."""
        return preprocess_w600k_crop(face_crop, target_size=(112, 112), is_bgr=True)

    def embed(self, face_crop: np.ndarray) -> Optional[FaceEmbedding]:
        """
        Extracts a 512-D L2-normalized identity embedding from a face crop.

        Returns:
            FaceEmbedding object if successful, None otherwise.
        """
        input_tensor = self.preprocess(face_crop)
        if input_tensor is None:
            return None

        # Execute ONNX Runtime inference
        ort_outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )

        raw_vector = ort_outputs[0].flatten().astype(np.float32)

        # Validate finiteness
        if np.isnan(raw_vector).any() or np.isinf(raw_vector).any():
            return None

        # L2 normalize embedding vector: v / ||v||_2
        norm = float(np.linalg.norm(raw_vector))
        if norm < 1e-12:
            return None

        normalized_vector = raw_vector / norm

        return FaceEmbedding(
            vector=normalized_vector,
            dimension=len(normalized_vector),
        )
