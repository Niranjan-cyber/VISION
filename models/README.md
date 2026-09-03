# VISION Model Weights & Storage

This directory holds deep learning model weights used by the actual pipeline
(`src/detection`, `src/face`, `src/anpr`). All are gitignored (see `.gitignore`)
and either auto-download on first run or must be supplied locally — see the
project root [README.md](../README.md#3-model-setup) for the authoritative,
up-to-date setup instructions.

## Models actually used by this pipeline

1. **YOLO11n** (`yolo11n.pt`, repo root, not `models/`):
   - Used by: [src/detection/detector.py](../src/detection/detector.py)
   - Purpose: person/vehicle detection (person, car, truck, bus, motorcycle, bicycle).
   - Auto-downloaded by `ultralytics` on first run if missing.

2. **YuNet** (`models/face_detection_yunet_2023mar.onnx`):
   - Used by: [src/face/detector.py](../src/face/detector.py)
   - Purpose: face detection + 5-point landmarks.
   - Auto-downloaded from the OpenCV Zoo on first run if missing.

3. **InsightFace W600K-R50** (`models/w600k_r50.onnx`):
   - Used by: [src/face/modern_embedder.py](../src/face/modern_embedder.py)
   - Purpose: 512-D face recognition embeddings.
   - Auto-downloaded from the official InsightFace `buffalo_l` release pack on first run if missing.

## Legacy / diagnostic-only models present in this folder

`arcface_resnet100*.onnx`, `insightface_w600k_r50.onnx`, `face_recognition_sface_2021dec.onnx`,
and `buffalo_l.zip` are leftovers from earlier model evaluation (see
`docs/PROJECT_SUMMARY.md` Slice 5.1–5.5 for why the ResNet100 ArcFace checkpoint
was abandoned) and from `src/face/*_diagnostic.py` scripts. They are **not**
required to run the pipeline or the demo — only `w600k_r50.onnx` (via
`--face-model w600k_r50`, the default) is used in production.
