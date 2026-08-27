# VISION — Comprehensive Project Progress Summary

**Platform:** VISION (AI-Powered Intelligent Border Video Analytics Platform)  
**Repository:** [https://github.com/Niranjan-cyber/VISION.git](https://github.com/Niranjan-cyber/VISION.git)  
**Active Branch:** `dev` (Upstream tracked)  
**Status:** ✅ **Vertical Slices 1, 2, & 3 Fully Completed, Tested, Audited & Pushed**

---

## 📌 Executive Overview

**VISION** is a modular, high-performance video analytics and surveillance intelligence platform designed for border monitoring. The system ingests conventional CCTV camera streams, runs real-time deep-learning object detection, performs multi-object trajectory tracking (ByteTrack), extracts person crops for face detection (YuNet), and associates detected faces to persistent person track IDs—all while maintaining strict architectural boundaries and zero framework lock-in.

---

## 🚀 Milestone Progress Breakdown

```text
VIDEO STREAM
     │
     ▼
[VideoSource] (OpenCV Ingestion)
     │
     ▼
[YOLODetector] (YOLO11n Object Detection) ──► Detection[]
     │
     ▼
[ByteTrackTracker] (Kalman Filter + Hungarian Matching) ──► Track[]
     │
     ├───────────────────────────────────┐
     ▼                                   ▼
Vehicle Tracks                    Person Tracks (class_name == "person")
(car, truck, bus, etc.)                  │
                                         ▼
                                  Person Crops (frame[y1:y2, x1:x2])
                                         │
                                         ▼
                                  [FaceDetector] (YuNet ONNX) ──► FaceDetection[]
                                         │
                                         ▼
                                  Crop-to-Global Coordinate Conversion
                                         │
                                         ▼
                                  [associate_faces_to_tracks] (Spatial IoU)
                                         │
                                         ▼
                                  Face ↔ Person Track ID
                                         │
     ┌───────────────────────────────────┘
     ▼
[Visualization & HUD Annotations]
```

---

### 1. Vertical Slice 1: Video Ingestion & Object Detection
- **`src/ingestion/video.py`**: Created reusable `VideoSource` class wrapping OpenCV `cv2.VideoCapture`. Exposes FPS, dimensions, frame count, and sequential frame extraction with zero YOLO coupling.
- **`src/detection/detector.py`**: Created `YOLODetector` class wrapping `yolo11n.pt`. Filters COCO target surveillance classes (`person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`).
- **`src/core/types.py`**: Introduced `BoundingBox` and `Detection` dataclasses to separate domain models from Ultralytics framework internals.
- **`src/main.py`**: Implemented CLI (`argparse`), frame sampling `--interval N`, glassmorphic dark HUD overlay (`Source FPS`, `Inference FPS`, `Frame count`, `Detections`), and end-of-stream summary reporting.

### 2. Vertical Slice 2: Multi-Object Tracking (ByteTrack)
- **`src/tracking/tracker.py`**: Implemented `ByteTrackTracker` wrapping Ultralytics' `BYTETracker` engine via a custom tensor adapter (`_DetectionResultsAdapter`). Performs Kalman filter trajectory estimation and 2-stage Hungarian algorithm matching (`lapx`).
- **`src/core/types.py`**: Added `Track` dataclass (`track_id`, `class_id`, `class_name`, `confidence`, `bbox`, `frame_number`).
- **`src/main.py`**: Rendered persistent track labels (`[class_name] #[track_id] [confidence]`), added `Active Tracks` counter to HUD, and output end-of-stream tracking statistics (`Unique Tracks`, `Max Active Tracks`).
- **`tests/test_tracking.py`**: Created automated unit test suite verifying track ID persistence across consecutive frames.

### 3. Vertical Slice 3: Face Detection & Person-Track Association
- **`src/core/types.py`**: Added `FaceDetection` dataclass (`bbox: BoundingBox`, `confidence: float`).
- **`src/face/detector.py`**: Created `FaceDetector` class using `cv2.FaceDetectorYN` with YuNet ONNX (`models/face_detection_yunet_2023mar.onnx`), running exclusively on person crops (`Track.class_name == "person"`).
- **`src/face/association.py`**: Created `FaceTrackAssociation` dataclass and `associate_faces_to_tracks` algorithm mapping full-frame face detections to person `track_id`s based on spatial containment and IoU.
- **`src/main.py`**: Implemented safe person crop extraction, crop-to-global coordinate conversion (`global_x = person_x + crop_x`), visualization (`face -> #17`), HUD counters (`Faces Detected`, `Faces Associated`), and summary reporting.
- **`tests/test_face.py`**: Added 7 comprehensive unit tests covering coordinate transformations, empty crops, spatial association, and vehicle exclusion.

---

## 📊 Benchmark & Verification Results

Executed against test video footage ([sample.mp4](file:///c:/Career/Hackathons/SIH/VISION/data/videos/sample.mp4)):

```text
==================================================
       VISION — Vertical Slice 3 Pipeline        
==================================================
 Video Path          : data/videos/sample.mp4
 YOLO Model          : yolo11n.pt
 Confidence Threshold: 0.25
 Face Confidence     : 0.50
 Inference Interval  : Every 1 frame(s)
==================================================
[INFO] Video loaded successfully. Resolution: 1280x720, FPS: 30.00, Total Frames: 911
[INFO] Reached end of video stream.
==================================================
VISION — Slice 3 Summary
==================================================
Frames Processed       : 911
YOLO Inferences        : 911
Total Detections       : 633
Unique Tracks          : 37
Max Active Tracks      : 3
Faces Detected         : 12
Faces Associated       : 12
Average Inference FPS  : ~23.97 FPS
==================================================
```

### Unit Test Suite Output:
```powershell
python -m unittest discover -s tests
```
```text
..........
----------------------------------------------------------------------
Ran 10 tests in 0.301s

OK
```

---

## 📂 Repository File Index

```text
VISION/
├── docs/
│   ├── PROJECT_SUMMARY.md       # Comprehensive progress summary
│   ├── VISION_PRD.md            # Product Requirements Document
│   ├── SIH_border_surveillance.md # Architecture reference
│   └── SLICE_1_SUMMARY.md       # Slice 1 report
│
├── data/
│   └── videos/
│       ├── .gitkeep
│       └── sample.mp4          # Test footage
│
├── models/
│   ├── .gitkeep
│   └── face_detection_yunet_2023mar.onnx # Pretrained YuNet face model
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── video.py           # VideoSource abstraction
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detector.py        # YOLODetector wrapper
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py         # ByteTrackTracker wrapper
│   │
│   ├── face/
│   │   ├── __init__.py
│   │   ├── detector.py        # FaceDetector ONNX wrapper
│   │   └── association.py     # FaceTrackAssociation & spatial IoU
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── types.py           # Domain dataclasses
│   │
│   └── main.py                # Pipeline controller & visualization HUD
│
├── tests/
│   ├── __init__.py
│   ├── test_tracking.py       # Tracking unit tests
│   └── test_face.py           # Face detection & association unit tests
│
├── .venv/                      # Isolated virtual environment
├── requirements.txt            # Dependencies (ultralytics, opencv-python, numpy, lapx)
├── .gitignore                 # Exclusions for binaries, models, & cache
└── README.md                  # Quick-start manual
```

---

## 🛠 Usage Instructions

### 1. Environment Activation
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 2. Run Test Suite
```powershell
python -m unittest discover -s tests
```

### 3. Launch VISION Pipeline
```powershell
python -m src.main --video data/videos/sample.mp4
```

---

## 🔮 Future Roadmap (Upcoming Vertical Slices)

- [x] **Vertical Slice 1:** Video Ingestion & YOLO11n Object Detection
- [x] **Vertical Slice 2:** Multi-Object Tracking using ByteTrack
- [x] **Vertical Slice 3:** Person-Crop Face Detection & Spatial Track Association
- [ ] **Vertical Slice 4:** Face Recognition & Embeddings (ArcFace / InsightFace 512-d feature extraction & matching)
- [ ] **Vertical Slice 5:** Video Enhancement Pipeline (Zero-DCE++ low-light, Real-ESRGAN super-resolution, Restormer deblurring)
- [ ] **Vertical Slice 6:** Virtual Fence Intrusion Detection & Threat Risk Engine
- [ ] **Slice 7:** Persistence Layer (PostgreSQL event logs & Redis transient trajectory caching)
- [ ] **Slice 8:** REST & WebSocket API Backend (FastAPI) + React Control Dashboard
