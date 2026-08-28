# VISION — Comprehensive Project Progress Summary

**Platform:** VISION (AI-Powered Intelligent Border Video Analytics Platform)  
**Repository:** [https://github.com/Niranjan-cyber/VISION.git](https://github.com/Niranjan-cyber/VISION.git)  
**Active Branch:** `main`  
**Status:** ✅ **Vertical Slices 1, 2, 3, 4, & 5 Fully Completed, Tested, Audited & Verified**

---

## 📌 Executive Overview

**VISION** is a modular, high-performance video analytics and surveillance intelligence platform designed for border monitoring. The system ingests conventional CCTV camera streams, runs real-time deep-learning object detection, performs multi-object trajectory tracking (ByteTrack), extracts person crops for face detection (YuNet), associates detected faces to persistent person track IDs, generates 512-dimensional L2-normalized feature embeddings using ArcFace, and matches embeddings against an in-memory reference face gallery via cosine similarity and configurable thresholding—all while maintaining strict architectural boundaries and zero framework lock-in.

---

## 🚀 Pipeline Architecture

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
                                  Associated Face Crops
                                         │
                                         ▼
                                  [preprocess_face_crop] (112x112 Normalized Blob)
                                         │
                                         ▼
                                  [FaceEmbedder] (ArcFace ResNet-100 ONNX)
                                         │
                                         ▼
                                  [l2_normalize] ──► FaceEmbedding (512 dimensions)
                                         │
                                         ▼
                                  [FaceGallery] (In-Memory Reference Vectors)
                                         │
                                         ▼
                                  [FaceMatcher] (Cosine Similarity)
                                         │
                                         ▼
                                  Threshold Check (similarity >= threshold)
                                     ↙       ↘
                               Known          Unknown
                                 │               │
     ┌───────────────────────────┴───────────────┘
     ▼
[Visualization & HUD Annotations]
```

---

## 📊 Live Verification Benchmark Results

### Benchmark 1: High-Resolution Surveillance Footage (`data/videos/sample1.mp4`)

Full HD 1080p footage processing test:

```text
==================================================
       VISION — Vertical Slice 5 Pipeline        
==================================================
 Video Path          : data/videos/sample1.mp4
 YOLO Model          : yolo11n.pt
 Confidence Threshold: 0.25
 Face Confidence     : 0.50
 Recognition Thresh  : 0.60
 Gallery Directory   : data/face_gallery
 Inference Interval  : Every 1 frame(s)
==================================================
[INFO] Video loaded successfully. Resolution: 1920x1080, FPS: 25.00, Total Frames: 611
[INFO] Reached end of video stream.
==================================================
VISION — Slice 5 Summary
==================================================
Frames Processed       : 611
YOLO Inferences        : 611
Total Detections       : 2443
Unique Tracks          : 40
Max Active Tracks      : 7
Faces Detected         : 671
Faces Associated       : 671
Embeddings Generated   : 671
Recognized Faces       : 0
Unknown Faces          : 671
Recognition Threshold  : 0.60
Average Inference FPS  : 2.86
==================================================
```

### Benchmark 2: Standard Footage (`data/videos/sample.mp4`)

```text
==================================================
VISION — Slice 5 Summary
==================================================
Frames Processed       : 911
YOLO Inferences        : 911
Total Detections       : 633
Unique Tracks          : 37
Max Active Tracks      : 3
Faces Detected         : 12
Faces Associated       : 12
Embeddings Generated   : 12
Recognized Faces       : 0
Unknown Faces          : 12
Recognition Threshold  : 0.60
Average Inference FPS  : 23.97
==================================================
```

---

## 🚀 Milestone Progress Breakdown

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
- **`tests/test_face.py`**: Added 7 unit tests covering coordinate transformations, empty crops, spatial association, and vehicle exclusion.

### 4. Vertical Slice 4: Face Embedding Generation (ArcFace 512-d)
- **`src/core/types.py`**: Added `FaceEmbedding` dataclass (`vector: np.ndarray`, `dimension: int = 512`).
- **`src/face/preprocessing.py`**: Created `preprocess_face_crop` (scaling & resizing face crops to 112x112) and `l2_normalize` (safe unit L2 normalization).
- **`src/face/embedder.py`**: Created `FaceEmbedder` wrapping ArcFace ResNet-100 ONNX (`models/arcface_resnet100.onnx`). Generates normalized 512-dimensional feature vector embeddings.
- **`src/main.py`**: Integrated face crop extraction for associated faces only, ArcFace inference, visualization indicators (`embedding ✓`), HUD counters (`Embeddings Generated`, `Embedding Dimension: 512`), and summary reporting.
- **`tests/test_embeddings.py`**: Added 8 unit tests validating 512-d output contract, L2 norm unit length, zero vector safety, full-frame crop extraction, and domain decoupling.

### 5. Vertical Slice 5: Face Recognition & Identity Matching
- **`src/core/types.py`**: Added `IdentityMatch` dataclass (`identity: Optional[str]`, `similarity: float`, `is_match: bool`).
- **`src/face/gallery.py`**: Implemented `FaceGallery` (in-memory container storing 512-d reference vectors per identity) and `load_gallery_from_dir` (scans `data/face_gallery/<identity>/*.jpg` to enroll identities).
- **`src/face/matcher.py`**: Implemented `FaceMatcher` (cosine similarity matcher via dot product on L2-normalized vectors against threshold).
- **`src/main.py`**: Integrated track-level identity caching (`track_id -> IdentityMatch`), face visualization (`face -> #17 | person_001 (0.86)` vs `face -> #17 | Unknown (0.43)`), HUD counters (`Recognized Faces`, `Unknown Faces`, `Recog Threshold`), CLI option `--face-threshold` (default `0.60`), and end-of-stream summary.
- **`tests/test_face_matching.py`**: Added 12 unit tests validating 512-d gallery enrollment, empty gallery safety, thresholding, similarity scoring, and domain decoupling.

---

## 🧪 Automated Unit Test Suite

Run the full automated test suite:

```powershell
python -m unittest discover -s tests
```

Output:
```text
..............................
----------------------------------------------------------------------
Ran 30 tests in 0.845s

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
│   ├── face_gallery/           # Reference identity images directory
│   │   └── .gitkeep
│   └── videos/
│       ├── .gitkeep
│       ├── sample.mp4          # 720p Test footage
│       └── sample1.mp4         # 1080p Full HD Test footage
│
├── models/
│   ├── .gitkeep
│   ├── face_detection_yunet_2023mar.onnx # Pretrained YuNet face model
│   └── arcface_resnet100.onnx           # Pretrained ArcFace ResNet-100 model
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
│   │   ├── association.py     # FaceTrackAssociation & spatial IoU
│   │   ├── preprocessing.py   # Crop scaling & L2 normalization
│   │   ├── embedder.py        # FaceEmbedder ArcFace ONNX wrapper
│   │   ├── gallery.py         # FaceGallery & load_gallery_from_dir
│   │   └── matcher.py         # FaceMatcher cosine similarity search
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
│   ├── test_face.py           # Face detection & association unit tests
│   ├── test_embeddings.py     # ArcFace embedding unit tests
│   └── test_face_matching.py   # Identity matching & gallery unit tests
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
# Run with face recognition threshold 0.60
python -m src.main --video data/videos/sample.mp4 --face-threshold 0.60
```

---

## ⚠️ Known Limitations

- **Prototype Recognition Threshold**: Default threshold (`0.60`) is a prototype threshold requiring calibration against domain-specific validation datasets.
- **In-Memory Gallery**: Gallery reference vectors are held in Python memory (`FaceGallery`). No persistent database (PostgreSQL / Redis / pgvector / FAISS) is used in Slice 5.
- **Image Alignment & Quality**: Recognition quality depends directly on lighting, resolution, and pose of the face crop extracted from video feed.

---

## 🔮 Roadmap: Future Vertical Slices

- [x] **Vertical Slice 1:** Video Ingestion & YOLO11n Object Detection
- [x] **Vertical Slice 2:** Multi-Object Tracking using ByteTrack
- [x] **Vertical Slice 3:** Person-Crop Face Detection & Spatial Track Association
- [x] **Vertical Slice 4:** ArcFace 512-d Face Embedding Generation
- [x] **Vertical Slice 5:** Face Identity Matching & In-Memory Gallery Search
- [ ] **Vertical Slice 6:** Video Enhancement Pipeline (Zero-DCE++ low-light, Real-ESRGAN super-resolution, Restormer deblurring)
- [ ] **Vertical Slice 7:** Virtual Fence Intrusion Detection & Threat Risk Engine
- [ ] **Vertical Slice 8:** Persistence Layer (PostgreSQL event logs & Redis transient trajectory caching)
- [ ] **Vertical Slice 9:** REST & WebSocket API Backend (FastAPI) + React Control Dashboard
