import os
import sys
import urllib.request
from typing import Optional
import cv2
import numpy as np

from src.core.types import FaceEmbedding
from src.face.preprocessing import l2_normalize, preprocess_face_crop

ARCFACE_MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/"
    "arcface/model/arcfaceresnet100-8.onnx"
)


class FaceEmbedder:
    """Pretrained ArcFace face recognition model wrapper producing 512-d L2-normalized embeddings."""

    TARGET_DIMENSION = 512

    def __init__(
        self,
        model_path: str = "models/arcface_resnet100.onnx",
    ):
        self.model_path = model_path
        self.net = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads ArcFace ONNX model, downloading if not found locally."""
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)

        if not os.path.exists(self.model_path):
            print(
                f"[INFO] ArcFace model not found locally at '{self.model_path}'. Downloading...",
                file=sys.stderr,
            )
            try:
                urllib.request.urlretrieve(ARCFACE_MODEL_URL, self.model_path)
                print(
                    f"[INFO] Successfully downloaded ArcFace model to '{self.model_path}'.",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"[ERROR] Failed to download ArcFace model: {e}", file=sys.stderr)
                raise RuntimeError(
                    f"Unable to download ArcFace model from {ARCFACE_MODEL_URL}"
                ) from e

        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
        except Exception as e:
            print(f"[ERROR] Failed to load ArcFace ONNX model: {e}", file=sys.stderr)
            raise RuntimeError(f"ArcFace model loading error: {e}") from e

    def embed(self, face_crop: np.ndarray) -> Optional[FaceEmbedding]:
        """
        Extracts a 512-dimensional L2-normalized face feature embedding from a BGR face crop.
        Returns None if face_crop is invalid, empty, or contains NaNs/Infs.
        Guarantees that each returned FaceEmbedding holds a distinct, independent NumPy vector.
        """
        if self.net is None or face_crop is None or not isinstance(face_crop, np.ndarray) or face_crop.size == 0:
            return None

        if np.isnan(face_crop).any() or np.isinf(face_crop).any():
            return None

        blob = preprocess_face_crop(face_crop)
        if blob is None:
            return None

        try:
            self.net.setInput(blob)
            raw_output = self.net.forward()
        except Exception as e:
            print(f"[WARNING] ArcFace inference failed: {e}", file=sys.stderr)
            return None

        if raw_output is None or raw_output.size == 0:
            return None

        raw_vec = raw_output.flatten()
        if len(raw_vec) != self.TARGET_DIMENSION:
            print(
                f"[ERROR] ArcFace embedding dimension mismatch: expected {self.TARGET_DIMENSION}, got {len(raw_vec)}",
                file=sys.stderr,
            )
            raise ValueError(
                f"Model produced {len(raw_vec)}-d embedding, expected {self.TARGET_DIMENSION}-d"
            )

        if np.isnan(raw_vec).any() or np.isinf(raw_vec).any():
            print("[WARNING] ArcFace raw vector contains NaNs or Infs. Rejecting crop.", file=sys.stderr)
            return None

        norm_vec = l2_normalize(raw_vec)

        # Return a fresh, independent FaceEmbedding dataclass instance
        return FaceEmbedding(vector=np.copy(norm_vec), dimension=self.TARGET_DIMENSION)
