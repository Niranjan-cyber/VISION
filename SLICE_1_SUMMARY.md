# VISION — Vertical Slice 1 Achievement Summary

**Project:** VISION (AI-Powered Intelligent Border Video Analytics Platform)  
**Slice:** Vertical Slice 1 — Core Ingestion & Real-Time Object Detection  
**Status:** ✅ **Fully Implemented, Tested & Verified**

---

## 🎯 Executive Summary

Vertical Slice 1 establishes the fundamental vision processing pipeline for **VISION**. It proves that the system can ingest conventional HD CCTV video feeds, run deep-learning object detection using **YOLO11n**, isolate key surveillance target classes, overlay real-time status HUD metrics, and generate end-of-stream detection analytical summaries.

---

## 📊 Live Verification Benchmark Results

The pipeline was executed and validated against actual test video footage (`data/videos/sample.mp4`):

```text
==================================================
       VISION — Vertical Slice 1 Pipeline        
==================================================
 Video Path          : data/videos/sample.mp4
 Model               : yolo11n.pt
 Confidence Threshold: 0.25
 Inference Interval  : Every 1 frame(s)
==================================================
[INFO] Video loaded successfully. Resolution: 1280x720, FPS: 30.00, Total Frames: 911
[INFO] Reached end of video stream.
==================================================
VISION — Detection Summary
==================================================
Frames Processed : 911
YOLO Inferences  : 911
Total Detections : 633

Detection Classes:
  person     : 5
  bicycle    : 0
  car        : 627
  motorcycle : 0
  bus        : 0
  truck      : 1

Average Inference FPS : ~23.97 FPS
==================================================
```

### Key Performance Highlights:
- **Video Input:** HD 1280x720 @ 30 FPS (911 total frames processed continuously).
- **Detection Rate:** 633 total surveillance targets detected across frames.
- **Inference Performance:** Real-time execution reaching up to **23.97 FPS**.

---

## 🏗 Architecture & Design System

The repository follows a clean, decoupled design with strict separation of concerns:

```text
VISION/
│
├── data/
│   └── videos/
│       ├── .gitkeep
│       └── sample.mp4         # Sample video footage
│
├── models/
│   └── .gitkeep               # YOLO model weights storage
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── video.py          # OpenCV VideoSource abstraction
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detector.py       # YOLODetector wrapper (YOLO11n)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── types.py          # BoundingBox & Detection domain types
│   │
│   └── main.py               # CLI controller, frame sampling & HUD
│
├── .venv/                     # Isolated local Python virtual environment
├── requirements.txt           # Minimal dependencies (ultralytics, opencv-python, numpy)
├── .gitignore                # Git exclusions
├── README.md                 # Project setup and user guide
└── SLICE_1_SUMMARY.md        # Vertical Slice 1 summary report
```

### Module Responsibilities:

| Component | File Path | Responsibilities |
| :--- | :--- | :--- |
| **Ingestion** | `src/ingestion/video.py` | Reusable `VideoSource` class wrapping OpenCV `cv2.VideoCapture`. Exposes FPS, dimensions, frame count, and current frame index. **Zero coupling** to YOLO or detection logic. |
| **Domain Types** | `src/core/types.py` | `BoundingBox` and `Detection` dataclasses. Decouples VISION from Ultralytics framework internals so model providers can be swapped seamlessly. |
| **Detection** | `src/detection/detector.py` | `YOLODetector` class wrapping `yolo11n.pt`. Filters COCO targets for surveillance classes (`person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`). |
| **Pipeline Controller** | `src/main.py` | CLI using `argparse`, frame-sampling interval logic, HUD metrics overlay (`Source FPS`, `Inference FPS`, `Frame count`, `Detections`), and terminal summary generation. |

---

## ⚙️ Environment & Isolation

- **Virtual Environment:** Configured at `.venv/`.
- **Dependencies:** Strictly limited to `ultralytics>=8.3.0`, `opencv-python>=4.8.0`, and `numpy>=1.24.0`.
- **Git Safety:** Excludes `.venv/`, `__pycache__/`, `*.pyc`, `*.pt`, and video binaries.

---

## 🎮 CLI Usage

To re-run the pipeline on any sample footage:

```powershell
# Activate Environment
.\.venv\Scripts\Activate.ps1

# Run Slice 1 Pipeline
python -m src.main --video data/videos/sample.mp4
```

### Optional Command Flags:
- `--confidence`: Set threshold (default `0.25`).
- `--interval`: Run inference every Nth frame for performance tuning (default `1`).
- `--model`: Pass custom YOLO model checkpoint (default `yolo11n.pt`).

---

## 🛡️ Scope Discipline & Future Roadmap

Vertical Slice 1 strictly fulfilled all Slice 1 acceptance criteria without premature over-engineering:
- ❌ No tracking (ByteTrack)
- ❌ No databases (PostgreSQL, Redis, pgvector)
- ❌ No backend/frontend frameworks (FastAPI, React)
- ❌ No complex message queues or Docker containers

**Next Slice (Vertical Slice 2):**  
Incorporating Multi-Object Tracking (ByteTrack) to track individual object trajectories and bottom-center spatial coordinates over time.
