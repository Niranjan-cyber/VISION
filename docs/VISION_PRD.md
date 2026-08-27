# VISION
## Intelligent Border Video Analytics Platform

**Document Type:** Product Requirements Document  
**Purpose:** Hackathon MVP / Working Prototype  
**Status:** Prototype Baseline  
**Primary Goal:** Transform existing CCTV infrastructure into an AI-enabled surveillance system through software-defined video analytics.

---

# 1. Executive Summary

VISION is an AI-driven video analytics platform designed to transform conventional CCTV infrastructure into an intelligent surveillance network without requiring dedicated smart-camera, Facial Recognition System (FRS), Automatic Number Plate Recognition (ANPR), or proprietary analytics hardware.

VISION ingests video from existing IP cameras, including RTSP streams, and applies computer vision models to extract actionable intelligence such as:

- Human detection and tracking
- Vehicle detection and classification
- Face detection
- Face recognition
- Automatic Number Plate Recognition
- Virtual-fence intrusion detection
- Movement detection
- Suspicious-activity indicators
- Night-time movement detection
- Real-time alert generation
- Event logging and historical analysis

The hackathon MVP will focus on establishing the complete intelligence pipeline:

**Video → Detection → Tracking/Recognition → Event Intelligence → Alert → Dashboard**

The architecture is intentionally designed around pretrained models and a shared perception layer rather than training an independent model for every surveillance capability.

---

# 2. Problem Statement

Border security forces deploy CCTV cameras at Border Out Posts (BOPs), check posts, border roads, and other strategic installations.

Conventional CCTV systems primarily provide:

- Video recording
- Live video monitoring
- Playback

They still require continuous human observation to identify security-relevant events.

Advanced capabilities such as:

- Facial Recognition
- ANPR
- Intrusion detection
- Object tracking
- Behavioral analytics

often require specialized cameras, dedicated processing hardware, or proprietary software.

This creates several problems:

1. High deployment cost
2. Hardware dependency
3. Difficult deployment in remote locations
4. Limited scalability
5. High dependence on human operators
6. Slow identification of security incidents
7. Difficulty integrating intelligence from different cameras

VISION addresses this by adding an AI analytics layer **on top of existing CCTV infrastructure**.

---

# 3. Product Vision

VISION should become a software-defined surveillance platform where:

> Any compatible CCTV stream can become an intelligent camera through centralized AI-powered video analytics.

The system should separate:

**Perception → Tracking → Recognition → Event Intelligence → Alerting**

This allows new capabilities to be added without redesigning the entire system.

---

# 4. Core Requirements

## 4.1 Video Ingestion

VISION shall support:

- MP4/video files for prototype testing
- Webcam input
- RTSP streams
- Standard IP CCTV cameras

The ingestion layer shall:

- Decode the stream
- Extract frames
- Maintain timestamps
- Handle frame-rate sampling
- Resize frames for inference
- Detect stream failures
- Expose camera status

---

## 4.2 Human Detection and Tracking

VISION shall detect humans using a pretrained object detector.

The system shall:

- Detect people
- Assign confidence scores
- Generate bounding boxes
- Track people across frames
- Assign persistent track IDs
- Maintain object trajectories

Example:

```text
Person #17
Person #23
Person #31
```

---

## 4.3 Vehicle Detection and Classification

VISION shall detect common vehicle classes including:

- Car
- Motorcycle
- Bus
- Truck
- Bicycle

The initial detector will use pretrained COCO classes.

Domain-specific vehicle classes can be added through later fine-tuning.

---

## 4.4 Face Detection

VISION shall detect faces independently from the general object detector.

The face pipeline shall support:

```text
Frame
 ↓
Face Detection
 ↓
Face Bounding Box
 ↓
Face Alignment
```

SCRFD/InsightFace will be used for the MVP.

---

## 4.5 Face Recognition

VISION shall support identification of faces against an authorized face gallery.

Pipeline:

```text
Face
 ↓
Alignment
 ↓
Face Embedding
 ↓
Vector Similarity Search
 ↓
Known / Unknown
```

Face embeddings will be stored using PostgreSQL + pgvector.

The system shall return:

- Identity
- Similarity score
- Camera
- Timestamp
- Associated track ID

Recognition should be gated so that embeddings are not unnecessarily generated for the same tracked face on every frame.

---

## 4.6 ANPR

ANPR is part of the overall product requirements but will be implemented after the core MVP pipeline is stable.

Expected pipeline:

```text
Vehicle
 ↓
Plate Detection
 ↓
Plate Crop
 ↓
OCR
 ↓
Plate Normalization
 ↓
Vehicle Event
```

Potential prototype technologies include a YOLO-based plate detector plus PaddleOCR or LPRNet.

LPRNet is a lightweight end-to-end license-plate-recognition approach designed for real-time recognition.

---

## 4.7 Virtual Fence

Operators shall be able to configure a polygon or line representing a restricted area.

Example:

```text
             RESTRICTED ZONE

        ┌──────────────────────┐
        │                      │
        │       Person #17     │
        │            ↓         │
────────┼──────────────────────┼
        │    VIRTUAL FENCE     │
        │                      │
        └──────────────────────┘
```

When a tracked object crosses the configured boundary:

```text
INTRUSION EVENT
```

shall be generated.

---

## 4.8 Suspicious Activity

The MVP shall use a **rule-based temporal event engine**, rather than immediately training a large behavioral model.

Examples:

```text
Person
+
Restricted zone
+
Prolonged presence
→ Suspicious Activity
```

```text
Vehicle
+
Restricted zone
+
Extended stationary period
→ Suspicious Vehicle Event
```

This provides a working behavioral layer while keeping the architecture extensible to future ML-based activity recognition.

---

## 4.9 Night-Time Movement

The MVP will initially use a combination of:

- Time/context
- Motion
- Object detection
- Tracking

rather than a dedicated night-activity model.

Future versions can introduce low-light enhancement and specialized night-time detection models.

---

## 4.10 Real-Time Alerts

VISION shall generate alerts for security events.

Example:

```text
🚨 HIGH PRIORITY

Intrusion Detected

Camera: BOP-03
Object: Person #17
Zone: Restricted Area
Time: 14:32:17
```

Alerts shall be:

- Persisted
- Delivered in real time
- Visible on the dashboard
- Associated with a camera and event

---

# 5. Key Architectural Principle

VISION should **not train a separate model for every feature**.

Instead:

```text
                    VIDEO
                      │
                      ▼
              SHARED PERCEPTION
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Objects      Faces       Motion
          │           │
          ▼           ▼
       Tracking   Recognition
          │           │
          └─────┬─────┘
                ▼
        Unified Object State
                │
                ▼
        Event Intelligence
                │
                ▼
             Alerts
```

The models provide reliable low-level observations.

The event engine converts those observations into security intelligence.

---

# 6. MVP Architecture

```text
                         VISION MVP
                             │
                             ▼
                  ┌─────────────────────┐
                  │  CCTV / RTSP / MP4  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Video Ingestion     │
                  │ OpenCV + FFmpeg     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Frame Processor     │
                  │ Resize / Sampling   │
                  └──────────┬──────────┘
                             │
               ┌─────────────┴─────────────┐
               │                           │
               ▼                           ▼
       ┌────────────────┐          ┌────────────────┐
       │ YOLO           │          │ SCRFD          │
       │ Object Detect  │          │ Face Detect    │
       └───────┬────────┘          └───────┬────────┘
               │                           │
               ▼                           ▼
       ┌────────────────┐          ┌────────────────┐
       │ ByteTrack      │          │ ArcFace        │
       │ Object Tracking│          │ Face Embedding │
       └───────┬────────┘          └───────┬────────┘
               │                           │
               └─────────────┬─────────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Unified Object State│
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Event Engine        │
                  │                     │
                  │ Virtual Fence       │
                  │ Intrusion           │
                  │ Movement            │
                  │ Identity Rules      │
                  └──────────┬──────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
              ┌──────────┐      ┌─────────────┐
              │ Redis    │      │ PostgreSQL  │
              │ Runtime  │      │ + pgvector  │
              │ State    │      │ Persistence │
              └────┬─────┘      └──────┬──────┘
                   │                   │
                   └─────────┬─────────┘
                             ▼
                      ┌─────────────┐
                      │ FastAPI     │
                      │ REST + WS   │
                      └──────┬──────┘
                             ▼
                      ┌─────────────┐
                      │ React       │
                      │ Dashboard   │
                      └─────────────┘
```

---

# 7. Architecture Components

## 7.1 Video Ingestion

**Technologies:**

- OpenCV
- FFmpeg

Responsibilities:

- Connect to video sources
- Decode frames
- Handle RTSP
- FPS sampling
- Frame timestamps
- Stream health

---

## 7.2 Object Detection

**Primary model:** YOLO11n initially.

Ultralytics provides pretrained YOLO11 checkpoints ranging from nano to extra-large, with COCO-pretrained detection models. YOLO11n is a sensible first prototype because of its small compute footprint; YOLO11s/m can be evaluated later if accuracy is insufficient.

Recommended progression:

```text
YOLO11n
   ↓
Evaluate
   ↓
YOLO11s
   ↓
YOLO11m if GPU allows
```

The latest Ultralytics documentation also lists newer YOLO26 models, but for the hackathon prototype YOLO11 is retained as the mature, already-selected baseline.

---

## 7.3 Object Tracking

**Model/algorithm:** ByteTrack.

ByteTrack is a tracking-by-detection approach that associates high- and lower-confidence detections to reduce fragmented trajectories. The original paper reports strong MOT17 results and real-time performance.

Pipeline:

```text
YOLO detections
      ↓
ByteTrack
      ↓
Track IDs
      ↓
Object trajectories
```

---

## 7.4 Face Detection

**Model:** SCRFD through InsightFace.

SCRFD is designed specifically for efficient face detection under compute constraints and is appropriate for a real-time surveillance prototype.

For the easiest integration, use the InsightFace model package rather than implementing SCRFD from scratch.

---

## 7.5 Face Recognition

**Model:** ArcFace-based InsightFace recognition model.

The ArcFace paper introduces additive angular margin loss to improve the discriminative power of face embeddings.

Pipeline:

```text
SCRFD
 ↓
Face crop
 ↓
Alignment
 ↓
ArcFace
 ↓
Embedding
 ↓
pgvector
 ↓
Identity
```

InsightFace provides model packages such as `buffalo_l` containing a SCRFD detector and recognition model.

**Important licensing constraint:** InsightFace's code is MIT licensed, but its pretrained models/training data are subject to separate licensing terms; the public pretrained models are intended for non-commercial research unless separately licensed. This must be resolved before operational or commercial deployment.

---

# 8. Datasets

## 8.1 COCO

Primary object-detection baseline.

Useful classes:

```text
person
bicycle
car
motorcycle
bus
truck
```

COCO 2017 contains 123,272 train/validation images, 886,147 bounding boxes and 80 object classes.

Reference paper:

**Microsoft COCO: Common Objects in Context**.

---

## 8.2 WIDER FACE

Use for face-detection research and evaluation.

WIDER FACE contains challenging variation in:

- Scale
- Pose
- Occlusion
- Scene type

making it useful for evaluating face detectors under difficult conditions.

---

## 8.3 VGGFace2

Useful for understanding face-recognition research and robustness across pose and age.

The dataset contains approximately 3.31 million images covering 9,131 subjects.

For the MVP, **do not train ArcFace yourself**.

Use pretrained embeddings and create a small authorized demonstration gallery.

---

## 8.4 Domain-specific VISION dataset

After the baseline works, collect frames representative of the target deployment:

```text
BOP cameras
Border roads
Check posts
Long-distance people
Low-resolution cameras
Day
Night
Rain
Occlusion
Vehicles
Fences
```

This dataset becomes the basis for future fine-tuning.

The strategy is:

```text
Pretrained model
      ↓
Evaluate on VISION footage
      ↓
Identify failure cases
      ↓
Annotate failures
      ↓
Fine-tune
      ↓
Re-evaluate
```

Do not train from scratch.

---

# 9. Research Papers / Technical References

## Object Detection

**YOLO / Ultralytics documentation**

Use the implementation documentation and pretrained checkpoints as the practical reference. YOLO11 provides detection checkpoints from nano through extra-large.

---

## Object Tracking

**ByteTrack: Multi-Object Tracking by Associating Every Detection Box**

ECCV 2022.

Use this to understand tracking-by-detection and persistent object identities.

---

## Face Detection

**WIDER FACE: A Face Detection Benchmark**

Use this as the primary face-detection dataset/research reference.

---

## Face Recognition

**ArcFace: Additive Angular Margin Loss for Deep Face Recognition**

CVPR 2019.

This is the key research foundation for the face-embedding approach used by the prototype.

---

## Face Recognition Dataset

**VGGFace2: A dataset for recognising faces across pose and age**

Use it as a research/reference dataset rather than rebuilding the recognition model for the MVP.

---

## ANPR

**LPRNet: License Plate Recognition via Deep Neural Networks**

Use this as a research reference for lightweight end-to-end plate recognition.

---

# 10. Data Architecture

## Redis

Redis is used for **temporary, high-frequency state**.

Examples:

```text
camera:{id}:status

camera:{id}:latest_detections

camera:{id}:active_tracks

camera:{id}:fps

camera:{id}:recent_events
```

Redis answers:

> What is happening right now?

---

## PostgreSQL

PostgreSQL is the persistent source of truth.

Core entities:

```text
cameras
persons
detections
tracks
faces
events
alerts
```

---

## pgvector

Face embeddings are stored inside PostgreSQL.

```text
persons
   │
   └── face_embeddings
           │
           └── vector
```

This avoids introducing a separate vector database for the MVP.

---

# 11. Core Data Model

```text
Camera
 ├── id
 ├── name
 ├── source_url
 ├── location
 └── status

Person
 ├── id
 ├── name
 └── external_id

Detection
 ├── id
 ├── camera_id
 ├── track_id
 ├── class
 ├── confidence
 ├── bbox
 └── timestamp

Face
 ├── id
 ├── camera_id
 ├── track_id
 ├── person_id
 ├── confidence
 ├── embedding
 └── timestamp

Event
 ├── id
 ├── camera_id
 ├── event_type
 ├── track_id
 ├── severity
 ├── metadata
 └── timestamp

Alert
 ├── id
 ├── event_id
 ├── severity
 ├── status
 └── timestamp
```

---

# 12. Suggested Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python | AI/backend |
| Video | OpenCV | Frame ingestion |
| Video codec | FFmpeg | RTSP/video decoding |
| Detection | YOLO11n | Person/vehicle detection |
| Tracking | ByteTrack | Object tracking |
| Face detection | SCRFD | Face localization |
| Face recognition | ArcFace/InsightFace | Face embeddings |
| Vector search | pgvector | Face matching |
| Database | PostgreSQL | Persistent storage |
| Cache | Redis | Temporary state |
| Backend | FastAPI | REST API |
| Realtime | WebSocket | Live alerts |
| Frontend | React + TypeScript | Dashboard |
| Styling | TailwindCSS | UI |
| Containers | Docker Compose | Local deployment |
| GPU | NVIDIA CUDA | AI acceleration |

---

# 13. Suggested Development Tools

## Development

- VS Code
- Git
- GitHub
- Python 3.x
- Node.js
- Docker Desktop
- Postman/Insomnia

## AI/CV

- Ultralytics
- OpenCV
- InsightFace
- ONNX Runtime
- PyTorch
- NumPy

## Backend

- FastAPI
- Pydantic
- SQLAlchemy
- Alembic
- psycopg

## Frontend

- React
- TypeScript
- Vite
- TailwindCSS
- Recharts

## Infrastructure

- Docker Compose
- PostgreSQL
- pgvector
- Redis

---

# 14. Repository Structure

```text
VISION/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   │
│   │   ├── api/
│   │   │   ├── cameras.py
│   │   │   ├── events.py
│   │   │   ├── alerts.py
│   │   │   └── websocket.py
│   │   │
│   │   ├── ingestion/
│   │   │   ├── source.py
│   │   │   └── stream.py
│   │   │
│   │   ├── vision/
│   │   │   ├── detector.py
│   │   │   ├── tracker.py
│   │   │   ├── face.py
│   │   │   └── recognition.py
│   │   │
│   │   ├── events/
│   │   │   ├── engine.py
│   │   │   └── virtual_fence.py
│   │   │
│   │   ├── storage/
│   │   │   ├── postgres.py
│   │   │   └── redis.py
│   │   │
│   │   └── pipeline.py
│   │
│   └── requirements.txt
│
├── frontend/
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── services/
│       └── types/
│
├── models/
│
├── data/
│   ├── videos/
│   └── face_gallery/
│
├── configs/
│   ├── cameras.yaml
│   └── zones.yaml
│
├── docker-compose.yml
├── .env.example
└── README.md
```

---

# 15. MVP Scope

## Must Have

- MP4 ingestion
- Webcam ingestion
- RTSP-compatible ingestion
- Person detection
- Vehicle detection
- Object tracking
- Face detection
- Face recognition
- Authorized face gallery
- PostgreSQL
- pgvector
- Redis
- Virtual fence
- Intrusion events
- Real-time alerts
- Event logging
- FastAPI
- WebSocket
- React dashboard

## Prototype-Ready / Next Slice

- ANPR
- Vehicle watchlists
- Night movement
- Suspicious-activity rules

## Post-MVP

- Advanced behavioral models
- Multi-camera re-identification
- Edge inference
- Distributed GPU inference
- Command-and-control integration
- Advanced low-light enhancement
- Domain-specific model training

---

# 16. Vertical-Slice Implementation Roadmap

The system should be built **vertically**, not layer-by-layer.

---

## Slice 1 — Video → YOLO

### Goal

Prove that VISION can turn a conventional video into an AI-annotated video.

```text
MP4
 ↓
OpenCV
 ↓
Frame
 ↓
YOLO
 ↓
Bounding Boxes
 ↓
Display
```

### Deliverable

A working video with:

```text
Person
Car
Truck
Motorcycle
```

bounding boxes.

**No database. No frontend. No Redis.**

---

## Slice 2 — Object Tracking

Add ByteTrack.

```text
Video
 ↓
YOLO
 ↓
ByteTrack
 ↓
Persistent IDs
```

Output:

```text
Person #17
Person #21
Car #04
```

### Deliverable

Stable object IDs and trajectories.

---

## Slice 3 — Face Detection

Add SCRFD.

```text
Frame
 ↓
SCRFD
 ↓
Face Bounding Boxes
```

Now the same frame contains:

```text
Person #17
Car #04
Face #03
```

### Deliverable

Object + face detection running simultaneously.

---

## Slice 4 — Face Recognition

Add:

```text
SCRFD
 ↓
Alignment
 ↓
ArcFace
 ↓
Embedding
```

Create a small authorized face gallery.

```text
Person_001
Person_002
Person_003
```

### Deliverable

```text
Known Person
Unknown Person
```

with similarity scores.

---

## Slice 5 — PostgreSQL + pgvector

Persist:

- Cameras
- Detections
- Tracks
- Persons
- Faces
- Embeddings
- Events

### Deliverable

Restart the application and historical events remain available.

---

## Slice 6 — Redis

Add real-time state:

```text
Active tracks
Latest detections
Camera status
Recent events
```

### Deliverable

The system can efficiently expose current camera state without constantly querying PostgreSQL.

---

## Slice 7 — Virtual Fence

Add polygon/line configuration.

```text
Tracked object
 ↓
Position
 ↓
Zone check
 ↓
Boundary crossed?
 ↓
Intrusion event
```

### Deliverable

A person physically crossing the virtual boundary produces an event.

---

## Slice 8 — Event + Alert Engine

Create:

```text
Event
 ↓
Severity
 ↓
Alert
```

Example:

```text
INTRUSION
HIGH
BOP-03
Person #17
```

---

## Slice 9 — FastAPI + WebSocket

Expose:

```text
GET /cameras
GET /events
GET /alerts
GET /detections
```

and:

```text
/ws/events
```

for real-time alerts.

### Deliverable

The backend can feed the dashboard.

---

## Slice 10 — React Dashboard

Build:

- Live camera
- Detection overlays
- Face identity
- Camera status
- Alert panel
- Event timeline
- Detection statistics

### Deliverable

Complete end-to-end VISION demonstration.

---

# 17. ANPR Extension

After the core MVP:

```text
Vehicle Detection
       ↓
Plate Detection
       ↓
Plate Crop
       ↓
OCR / LPR
       ↓
Plate Number
       ↓
Watchlist Check
       ↓
Event
```

Potential starting technologies:

- YOLO for plate localization
- PaddleOCR
- LPRNet

LPRNet provides a lightweight end-to-end recognition approach and is specifically relevant to real-time ANPR research.

---

# 18. Performance Strategy

A critical design decision is **not to run every model on every frame**.

For example:

```text
30 FPS CCTV
     │
     ▼
Frame Scheduler
     │
     ├── YOLO → ~5–15 FPS
     │
     ├── ByteTrack → every available frame
     │
     └── Face Recognition → only when required
```

Face recognition should be gated:

```text
New face detected?
       │
      YES
       ↓
Generate embedding
       ↓
Search gallery
       ↓
Cache identity
```

The same tracked face should not require a fresh expensive recognition inference on every frame.

---

# 19. Success Criteria for the Hackathon

The MVP is successful if we can demonstrate:

### Scenario 1 — Person Detection

```text
CCTV
 ↓
Person detected
 ↓
Track ID assigned
```

### Scenario 2 — Vehicle

```text
CCTV
 ↓
Vehicle detected
 ↓
Vehicle class identified
 ↓
Track ID assigned
```

### Scenario 3 — Face

```text
CCTV
 ↓
Face detected
 ↓
Identity matched
```

### Scenario 4 — Intrusion

```text
Person
 ↓
Crosses virtual fence
 ↓
Event generated
 ↓
Alert generated
 ↓
Dashboard updates
```

### Scenario 5 — Persistence

```text
Event
 ↓
PostgreSQL
 ↓
Historical event visible
```

This demonstrates the complete product rather than isolated AI demos.

---

# 20. Final Product Architecture

The final conceptual model for VISION is:

```text
                         VISION
                           │
                           ▼
                    EXISTING CCTV
                           │
                           ▼
                    VIDEO INGESTION
                           │
                           ▼
                  SHARED PERCEPTION
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
           YOLO          SCRFD         Motion
             │             │
             ▼             ▼
        ByteTrack       ArcFace
             │             │
             └──────┬──────┘
                    ▼
             UNIFIED STATE
                    │
                    ▼
            EVENT INTELLIGENCE
                    │
       ┌────────────┼─────────────┐
       ▼            ▼             ▼
    INTRUSION    IDENTITY       ACTIVITY
       │            │             │
       └────────────┼─────────────┘
                    ▼
                  ALERT
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
       REDIS              POSTGRESQL
      Real-time          Persistent
        State              Events
          │                   │
          └─────────┬─────────┘
                    ▼
                 FASTAPI
                    │
               WebSocket
                    │
                    ▼
             VISION DASHBOARD
```

## 21. Core Design Philosophy

The most important architectural decision in VISION is:

> **Models detect primitives; the intelligence engine interprets them.**

We therefore avoid building a separate deep-learning model for every requirement.

Instead:

```text
YOLO
  → person / vehicle

ByteTrack
  → persistent identity in video

SCRFD
  → face

ArcFace
  → face embedding / identity

Motion + temporal rules
  → movement

Geometry + tracking
  → virtual fence

Event Engine
  → intrusion / suspicious events / alerts
```

This gives the hackathon prototype a clear path from **working MVP → domain fine-tuning → advanced surveillance intelligence**, without throwing away the initial architecture.

### Primary references

- [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11?utm_source=chatgpt.com) — pretrained detection models and implementation reference.
- [InsightFace repository](https://github.com/deepinsight/insightface?utm_source=chatgpt.com) — face detection/recognition implementation and model ecosystem.
- [ByteTrack paper](https://arxiv.org/abs/2110.06864?utm_source=chatgpt.com) — object tracking methodology.
- [ArcFace paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf?utm_source=chatgpt.com) — face-recognition embedding methodology.
- [COCO dataset paper](https://arxiv.org/abs/1405.0312?utm_source=chatgpt.com) — object-detection dataset foundation.
- [WIDER FACE paper](https://arxiv.org/abs/1511.06523?utm_source=chatgpt.com) — face-detection benchmark.
- [VGGFace2 paper](https://arxiv.org/abs/1710.08092?utm_source=chatgpt.com) — face-recognition dataset/reference.
- [LPRNet paper](https://arxiv.org/abs/1806.10447?utm_source=chatgpt.com) — lightweight ANPR recognition reference.