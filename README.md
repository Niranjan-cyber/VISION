# VISION — AI-Powered Border Surveillance & Video Analytics Platform

**VISION** is an AI-powered Intelligent Border Video Analytics Platform. This repository contains the core vision processing engine, supporting real-time video stream ingestion, object detection (YOLO11n), multi-object tracking (ByteTrack), facial landmark alignment (YuNet 5-point), and deep face recognition (InsightFace W600K-R50).

---

## 📚 Project Documentation

All detailed specifications, PRDs, slice summaries, diagnostic reports, and architecture overviews are organized in the [`docs/`](file:///c:/Career/Hackathons/SIH/VISION/docs) directory:

- 📄 **[PROJECT_SUMMARY.md](file:///c:/Career/Hackathons/SIH/VISION/docs/PROJECT_SUMMARY.md)**: Current project status, slice-by-slice history, benchmark metrics, and known issues.
- 📄 **[VISION_PRD.md](file:///c:/Career/Hackathons/SIH/VISION/docs/VISION_PRD.md)**: Complete Product Requirements Document.
- 📄 **[SIH_border_surveillance.md](file:///c:/Career/Hackathons/SIH/VISION/docs/SIH_border_surveillance.md)**: System design and technology selection ## 🔄 Current Pipeline Architecture (Vertical Slice 7.0)

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
     ├───────────────────────────────────┬───────────────────────────────────┐
     ▼                                   ▼                                   ▼
Person Tracks                       Vehicle Tracks                      Spatial Geometry
(class_name == "person")            (car, truck, bus, etc.)             (BoundingBox.bottom_center)
     │                                   │                                   │
     ▼                                   ▼                                   │
[FaceDetector] (YuNet ONNX)         [LicensePlateDetector]                   │
     │                                   │                                   │
     ▼                                   ▼                                   │
[align_face] (5-Pt Similarity)      [PlateEnhancer] + [PlateOCR]             │
     │                                   │                                   │
     ▼                                   ▼                                   │
[W600KR50Embedder] (512-D L2 Norm)  [clean_plate_text]                       │
     │                                   │                                   │
     ▼                                   ▼                                   │
[FaceMatcher] / [VectorDB]          [PlateTrackCache]                        │
     │                                   │                                   │
     └───────────────────────────────────┴───────────────────────────────────┘
                                         │
                                         ▼
                            [Unified ObjectState]
                                         │
                                         ▼
                               [Zone / Geometry Evaluation]
                                (configs/zones.yaml)
                                         │
                                         ▼
                              [EventEngine Rule Evaluation]
                                 ├── INTRUSION (HIGH)
                                 ├── UNKNOWN_PERSON_INTRUSION (HIGH)
                                 ├── LOITERING (MEDIUM)
                                 └── SUSPICIOUS_VEHICLE (MEDIUM)
                                         │
                                         ▼
                         [SecurityEvent] & [Operational Alert]
                                         │
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

### Standard Production Run with Surveillance Zones
```bash
python -m src.main --video data/videos/shreyas1.mp4 --zones configs/zones.yaml
```

### Vehicle Intelligence with ANPR
```bash
python -m src.main --video data/videos/sample.mp4 --ocr-engine heuristic --zones configs/zones.yaml
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
| `--zones` | Path to YAML surveillance zone configuration | `None` |
| `--loitering-duration` | Seconds inside restricted zone before triggering `LOITERING` | `30.0` |
| `--stationary-duration` | Seconds vehicle is stationary before triggering `SUSPICIOUS_VEHICLE` | `60.0` |
| `--movement-threshold` | Displacement in pixels below which an object is stationary | `15.0` |
| `--confidence` | Confidence threshold for YOLO person/vehicle detection | `0.25` |
| `--face-confidence`| Confidence threshold for YuNet face detector | `0.50` |
| `--face-threshold` | Cosine similarity threshold for face recognition | `0.60` |
| `--face-margin` | Minimum difference between top-1 and top-2 identity | `0.10` |
| `--face-model` | Recognition backbone (`w600k_r50` or `r100`) | `w600k_r50` |
| `--gallery-dir` | Directory containing enrolled identity subfolders | `data/face_gallery` |
| `--disable-anpr` | Disable automatic license plate recognition | `False` |
| `--ocr-engine` | ANPR OCR backend (`auto`, `easyocr`, `heuristic`, `mock`) | `auto` |

---

## 📊 Current Project Status & Milestones

### ✅ Completed Milestones (Slices 1 to 7.0)
- **Slice 1:** Modular video ingestion (`VideoSource`) & YOLO11n object detection.
- **Slice 2:** ByteTrack multi-object tracking with persistent track IDs.
- **Slice 3:** YuNet deep face detection with 5-point facial landmark estimation.
- **Slice 4:** Spatial IoU association linking face detections to person tracks.
- **Slice 5:** Face gallery architecture, cosine similarity matching, and track-level caching.
- **Slice 5.5–5.6:** Modern InsightFace W600K-R50 ONNX integration ($512$-D embedding space) with 5-point facial landmark similarity transformation alignment.
- **Slice 5.7:** Cross-environment diagnostic suite evaluating multi-video datasets, threshold sweeps, and temporal aggregation.
- **Slice 6.0:** ANPR Vehicle Intelligence pipeline (Plate detection, contrast enhancement, OCR abstraction, Indian syntax regex cleaning, temporal consensus cache, PostgreSQL vector persistence).
- **Slice 7.0:** Event Intelligence & Alert Engine (Unified `ObjectState`, YAML zone definitions, point-in-polygon ray-casting on ground coordinates, zone transition detection, intrusion, unknown-person intrusion, loitering, stationary vehicle rules, deduplication, and operational `Alert` representations).

### 🧪 Test Suite Status
All **129 unit and integration tests** pass synchronously:
```bash
python -m unittest discover -s tests
# Ran 129 tests — OK
```

---

## 🗺️ Roadmap & Next Slices

- [ ] **Slice 8.0:** FastAPI REST backend & WebSockets streaming for live camera feeds and security alerts.
- [ ] **Slice 9.0:** Modern React Dashboard & Real-Time Security Operations Center (SOC) UI.