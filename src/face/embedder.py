import os
import sys
import urllib.request
from typing import Optional
import cv2
import numpy as np
import onnxruntime as ort

from src.core.types import FaceEmbedding
from src.face.preprocessing import l2_normalize, preprocess_face_crop

ARCFACE_MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/"
    "arcface/model/arcfaceresnet100-8.onnx"
)


class OpenCVArcFaceEmbedder:
    """Legacy ArcFace face recognition model wrapper using OpenCV DNN backend."""

    TARGET_DIMENSION = 512

    def __init__(
        self,
        model_path: str = "models/arcface_resnet100.onnx",
    ):
        self.model_path = model_path
        self.net = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads ArcFace ONNX model via OpenCV DNN, downloading if not found locally."""
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
            print(f"[ERROR] Failed to load ArcFace ONNX model with OpenCV DNN: {e}", file=sys.stderr)
            raise RuntimeError(f"OpenCV ArcFace model loading error: {e}") from e

    def embed(self, face_crop: np.ndarray) -> Optional[FaceEmbedding]:
        """
        Extracts a 512-dimensional L2-normalized face feature embedding from a BGR face crop
        using OpenCV DNN inference.
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
            print(f"[WARNING] OpenCV ArcFace inference failed: {e}", file=sys.stderr)
            return None

        if raw_output is None or raw_output.size == 0:
            return None

        raw_vec = raw_output.flatten()
        if len(raw_vec) != self.TARGET_DIMENSION:
            raise ValueError(
                f"Model produced {len(raw_vec)}-d embedding, expected {self.TARGET_DIMENSION}-d"
            )

        if np.isnan(raw_vec).any() or np.isinf(raw_vec).any():
            print("[WARNING] OpenCV ArcFace raw vector contains NaNs or Infs. Rejecting crop.", file=sys.stderr)
            return None

        norm_vec = l2_normalize(raw_vec)
        return FaceEmbedding(vector=np.copy(norm_vec), dimension=self.TARGET_DIMENSION)


class ONNXRuntimeArcFaceEmbedder:
    """ArcFace face recognition model wrapper using ONNX Runtime backend."""

    TARGET_DIMENSION = 512
    EXPECTED_INPUT_SHAPE = (1, 3, 112, 112)

    def __init__(
        self,
        model_path: str = "models/arcface_resnet100.onnx",
    ):
        self.model_path = model_path
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = "data"
        self.output_name: str = "fc1"
        self.input_shape = list(self.EXPECTED_INPUT_SHAPE)
        self.output_shape = [1, self.TARGET_DIMENSION]
        self._load_model()

    def _load_model(self) -> None:
        """Loads ArcFace ONNX model via ONNX Runtime, downloading if necessary."""
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
            # Initialize ONNX Runtime InferenceSession with CPUExecutionProvider
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

            # Inspect model metadata dynamically
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()

            if not inputs:
                raise RuntimeError("ONNX model has no input nodes.")
            if not outputs:
                raise RuntimeError("ONNX model has no output nodes.")

            self.input_name = inputs[0].name
            self.input_shape = list(inputs[0].shape)
            self.output_name = outputs[0].name
            self.output_shape = list(outputs[0].shape)

            # Validate output dimension
            last_dim = self.output_shape[-1] if self.output_shape else None
            if last_dim != self.TARGET_DIMENSION:
                raise ValueError(
                    f"ArcFace ONNX model output dimension mismatch: expected {self.TARGET_DIMENSION}, got {last_dim}"
                )

        except Exception as e:
            print(f"[ERROR] Failed to load ArcFace ONNX model with ONNX Runtime: {e}", file=sys.stderr)
            raise RuntimeError(f"ONNX Runtime ArcFace model loading error: {e}") from e

    def preprocess(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocesses a BGR face crop into the normalized NCHW float32 tensor:
        HWC BGR -> BGR to RGB -> resize to 112x112 -> float32 -> (pixel - 127.5) / 128.0 -> HWC to CHW -> add batch dim -> [1, 3, 112, 112]
        """
        if face_crop is None or not isinstance(face_crop, np.ndarray) or face_crop.size == 0:
            return None

        if np.isnan(face_crop).any() or np.isinf(face_crop).any():
            return None

        h, w = face_crop.shape[:2]
        if h < 5 or w < 5:
            return None

        try:
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
            f_img = resized.astype(np.float32)
            normalized = (f_img - 127.5) / 128.0
            chw = np.transpose(normalized, (2, 0, 1))
            tensor = np.expand_dims(chw, axis=0)
            return np.ascontiguousarray(tensor, dtype=np.float32)
        except Exception:
            return None

    def embed(self, face_crop: np.ndarray) -> Optional[FaceEmbedding]:
        """
        Extracts a 512-dimensional L2-normalized face feature embedding from a BGR face crop
        using ONNX Runtime inference.
        Returns None if face_crop is invalid, empty, or inference fails.
        """
        if self.session is None or face_crop is None or not isinstance(face_crop, np.ndarray) or face_crop.size == 0:
            return None

        tensor = self.preprocess(face_crop)
        if tensor is None:
            return None

        try:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: tensor},
            )
        except Exception as e:
            print(f"[WARNING] ONNX Runtime ArcFace inference failed: {e}", file=sys.stderr)
            return None

        if not outputs or outputs[0] is None:
            return None

        raw_vec = outputs[0].flatten().astype(np.float32)

        if len(raw_vec) != self.TARGET_DIMENSION:
            raise ValueError(
                f"Model produced {len(raw_vec)}-d embedding, expected {self.TARGET_DIMENSION}-d"
            )

        if np.isnan(raw_vec).any() or np.isinf(raw_vec).any():
            print("[WARNING] ONNX Runtime ArcFace output contains NaNs or Infs. Rejecting crop.", file=sys.stderr)
            return None

        norm = float(np.linalg.norm(raw_vec))
        if norm < 1e-12 or np.isnan(norm) or np.isinf(norm):
            print("[WARNING] ArcFace raw embedding norm is zero or invalid. Rejecting crop.", file=sys.stderr)
            return None

        norm_vec = raw_vec / norm
        return FaceEmbedding(vector=np.copy(norm_vec), dimension=self.TARGET_DIMENSION)


# Default FaceEmbedder alias points to the production ONNXRuntime backend
FaceEmbedder = ONNXRuntimeArcFaceEmbedder
