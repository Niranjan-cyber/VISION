# VISION Model Weights & Storage

This directory holds deep learning models and weight checkpoints required for the vision pipeline.

## Required Models

1. **YOLO Detection Model** (`yolov8n.pt` / `yolov8x.pt`):
   - Used by: `backend/app/vision/detector.py`
   - Purpose: Multi-class object detection (Person, Car, Truck, Bus, Motorcycle, Bicycle).
   - Source: [Ultralytics YOLO](https://docs.ultralytics.com/)

2. **CCTV Enhancement Models** (Place under `models/enhancement/`):
   - **Zero-DCE++**: Low-light frame enhancement (`zero_dce.pt`)
   - **Real-ESRGAN / BasicVSR++**: Super-resolution enhancement for low-res frames (`RealESRGAN_x4plus.pth`)
   - **RVRT / Restormer**: Frame deblurring model (`restormer.pth`)

3. **Face Recognition Models**:
   - **SCRFD**: Face detection model (`scrfd_10g_bnkps.onnx`)
   - **ArcFace**: Face recognition & 512-d feature extraction (`glintr100.onnx`)
   - Source: [InsightFace](https://github.com/deepinsight/insightface)
