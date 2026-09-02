# VISION — AI-Powered Border Surveillance & Video Analytics Platform

**VISION** is an AI-powered Intelligent Border Video Analytics Platform. This repository contains the core vision processing engine, supporting real-time video stream ingestion, object detection (YOLO11n), multi-object tracking (ByteTrack), facial landmark alignment (YuNet 5-point), and deep face recognition (InsightFace W600K-R50).

---

## 📚 Project Documentation

All detailed specifications, PRDs, slice summaries, diagnostic reports, and architecture overviews are organized in the [`docs/`](file:///c:/Career/Hackathons/SIH/VISION/docs) directory:

- 📄 **[PROJECT_SUMMARY.md](file:///c:/Career/Hackathons/SIH/VISION/docs/PROJECT_SUMMARY.md)**: Current project status, slice-by-slice history, benchmark metrics, and known issues.
- 📄 **[VISION_PRD.md](file:///c:/Career/Hackathons/SIH/VISION/docs/VISION_PRD.md)**: Complete Product Requirements Document.
- 📄 **[SIH_border_surveillance.md](file:///c:/Career/Hackathons/SIH/VISION/docs/SIH_border_surveillance.md)**: System design and technology selection guide.

---

## 🔄 Current Pipeline Architecture (Vertical Slice 5.7)

```text
VIDEO STREAM (MP4 / RTSP / Webcam)
     │
     ▼
[VideoSource] (OpenCV Ingestion)
     │
     ▼
[YOLODetector] (YOLO11n Object Detection) ──► Detection[] (person, car, truck, bus, etc.)
     │
     ▼
[ByteTrackTracker] (Kalman Filter + 2-Stage Hungarian Matching) ──► Track[]
     │
     ├───────────────────────────────────┐
     ▼                                   ▼
Vehicle Tracks                    Person Tracks (class_name == "person")
(car, truck, bus, etc.)                  │
                                         ▼
                                  Person Crop [py1:py2, px1:px2]
                                         │
                                         ▼
                                  [FaceDetector] (YuNet ONNX) ──► 5 Facial Landmarks
                                         │
                                         ▼
                                  Crop-to-Global Coordinate Mapping
                                         │
                                         ▼
                                  [align_face] (5-Point Affine Similarity Transform)
                                         │
                                         ▼
                                  Canonical 112×112 Aligned BGR Crop
                                         │
                                         ▼
                                  [W600KR50Embedder] (InsightFace ResNet-50 ONNX)
                                         │
                                         ▼
                                  [l2_normalize] ──► 512-D L2-Normalized Embedding
                                         │
                                         ▼
                                  [FaceGallery] (Aligned Multi-Image Reference Vectors)
                                         │
                                         ▼
                                  [FaceMatcher] (Cosine Similarity + Margin Check)
                                         │
                                         ▼
                                  Track-Level Identity Cache (Instant O(1) Lookups)
                                     ↙       ↘
                                Known          Unknown
                                 │               │
     ┌───────────────────────────┴───────────────┘
     ▼
[Visualization & HUD Annotations]
```

---

## 🛠 Python Environment Setup

### 1. Create the Virtual Environment
```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment
* **Windows PowerShell:** `.\.venv\Scripts\Activate.ps1`
* **Windows Command Prompt (CMD):** `.\.venv\Scripts\activate.bat`
* **macOS / Linux:** `source .venv/bin/activate`

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

---

## 🚀 Running VISION

### Standard Production Run
```bash
python -m src.main --video data/videos/shreyas1.mp4 --face-threshold 0.60
```

### Diagnostic & Debug Modes
```bash
# Debug face recognition cosine scores & margins
python -m src.main --video data/videos/shreyas1.mp4 --face-threshold 0.60 --debug-face-matching

# Save canonical 112x112 aligned face crops
python -m src.main --video data/videos/shreyas1.mp4 --debug-face-alignment

# Run full cross-environment multi-video diagnostic suite
python -m src.face.cross_environment_diagnostic
```

### Command Line Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--video` | Path to input MP4 video file or RTSP stream | `data/videos/test.mp4` |
| `--confidence` | Confidence threshold for YOLO person/vehicle detection | `0.25` |
| `--face-confidence`| Confidence threshold for YuNet face detector | `0.50` |
| `--face-threshold` | Cosine similarity threshold for face recognition | `0.60` |
| `--face-margin` | Minimum difference between top-1 and top-2 identity | `0.10` |
| `--face-model` | Recognition backbone (`w600k_r50` or `r100`) | `w600k_r50` |
| `--gallery-dir` | Directory containing enrolled identity subfolders | `data/face_gallery` |
| `--debug-face-matching` | Print real-time similarity breakdown to terminal | `False` |
| `--debug-face-alignment`| Save first 5 aligned face crops to disk | `False` |

---

## 📊 Current Project Status & Diagnostic Findings

### ✅ Completed Milestones (Slices 1 to 5.7)
- **Slice 1:** Modular video ingestion (`VideoSource`) & YOLO11n object detection.
- **Slice 2:** ByteTrack multi-object tracking with persistent track IDs.
- **Slice 3:** YuNet deep face detection with 5-point facial landmark estimation.
- **Slice 4:** Spatial IoU association linking face detections to person tracks.
- **Slice 5:** Face gallery architecture, cosine similarity matching, and track-level caching.
- **Slice 5.1–5.4:** Diagnostic discovery of legacy ArcFace R100 feature collapse under normalized inputs.
- **Slice 5.5:** Integration of official InsightFace W600K-R50 ONNX model ($512$-D embedding space).
- **Slice 5.6:** 5-point facial landmark similarity transformation alignment (`align_face`) applied across both gallery enrollment and in-flight video frames.
- **Slice 5.7:** Cross-environment diagnostic suite evaluating multi-video datasets, face quality distributions, threshold sweeps, and temporal aggregation.

### 🧪 Test Suite Status
All **86 unit and integration tests** pass synchronously:
```bash
python -m unittest discover -s tests
# Ran 86 tests in ~20s — OK
```

---

## ⚠️ Current Issues & Diagnostic Insights (Slice 5.7)

Comprehensive empirical analysis across 5 video datasets (`shreyas1.mp4`, `jaysingpure1.mp4`, `jaysingpure2.mp4`, `atharva1.mp4`, `salman4.mp4`) revealed:

1. **Cross-Environment Threshold Gap (Case B)**:
   - High-resolution close-up footage (`shreyas1.mp4`, `jaysingpure1.mp4`) achieves genuine similarities of **`0.70 – 0.92`**.
   - Cross-environment CCTV footage (`atharva1.mp4`) achieves genuine similarities of **`0.40 – 0.52`** (Peak: `0.5143`).
   - Impostor scores on negative control footage (`salman4.mp4`) peak at **`0.1090`** (Mean: `0.0310`).
   - **Issue:** The fixed `0.60` threshold configured for close-up video causes CCTV genuine faces to be rejected despite having a $+0.40$ margin over impostors.
2. **Small Face Degradation at Long Distance**:
   - Faces $< 50\text{ px}$ (`jaysingpure2.mp4`) exhibit landmark instability, yielding low similarity scores ($< 0.25$).
3. **Limited Gallery Domain Variation**:
   - Current enrollment uses 3 studio reference photos per identity. Adding 5–10 diverse reference images per identity will bridge the ambient lighting domain gap.

---

## 🗺️ Roadmap & Next Slices

- [ ] **Slice 5.8 / 6.0:** Adaptive quality-aware recognition thresholding ($0.45 - 0.50$ for CCTV) and multi-frame temporal evidence pooling.
- [ ] **Slice 6.1:** Geo-fencing & restricted perimeter intrusion detection (virtual tripwires).
- [ ] **Slice 7.0:** FastAPI REST backend & WebSockets streaming for live camera feeds.
- [ ] **Slice 8.0:** Vector database integration (Qdrant / Milvus) for scaling face gallery to 100,000+ identities.