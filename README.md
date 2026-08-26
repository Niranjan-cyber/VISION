# VISION - Intelligent Border Surveillance & Vision Analytics Platform

**VISION** is an AI-powered border surveillance and video analytics framework designed for real-time CCTV stream ingestion, low-light/super-resolution video enhancement, object detection & tracking (YOLO + ByteTrack), virtual fence intrusion alerts, and face recognition (SCRFD + ArcFace).

---

## 📐 Project Repository Structure

```
VISION/
│
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI Application Entrypoint
│   │   │
│   │   ├── api/                 # REST & WebSocket API Routers
│   │   │   ├── cameras.py       # Camera feed management
│   │   │   ├── events.py        # Event query API
│   │   │   └── alerts.py        # Real-time alert broadcasts
│   │   │
│   │   ├── ingestion/           # Video Ingestion Layer
│   │   │   ├── source.py        # RTSP / IP stream frame capture
│   │   │   └── stream.py        # Async frame queue buffering
│   │   │
│   │   ├── vision/              # AI Core Models
│   │   │   ├── detector.py      # YOLO Object Detection
│   │   │   ├── tracker.py       # ByteTrack Multi-Object Tracking
│   │   │   ├── face.py          # SCRFD Face Detection
│   │   │   └── recognition.py   # ArcFace Face Recognition
│   │   │
│   │   ├── events/              # Risk & Security Logic
│   │   │   ├── engine.py        # Threat & Risk calculation engine
│   │   │   └── virtual_fence.py # Polygon intrusion detection
│   │   │
│   │   ├── storage/             # Data Persistence Layer
│   │   │   ├── postgres.py      # PostgreSQL permanent storage
│   │   │   └── redis.py         # Redis transient trajectory storage
│   │   │
│   │   └── pipeline.py          # End-to-end vision processing pipeline
│   │
│   ├── requirements.txt         # Python dependencies
│   └── Dockerfile               # Backend Docker image setup
│
├── frontend/                    # Web Interface
│   ├── src/
│   │   ├── components/          # UI Components
│   │   ├── pages/               # Dashboard pages
│   │   ├── services/            # API integration services
│   │   └── types/               # TypeScript interface definitions
│   ├── package.json             # Frontend dependencies
│   └── Dockerfile               # Frontend containerization
│
├── models/                      # Deep learning model weights directory
│   └── README.md                # Download & placement instructions
│
├── data/                        # Local data storage
│   ├── videos/                  # Sample test footage
│   └── face_gallery/            # Target identity gallery
│
├── configs/                     # System configurations
│   ├── cameras.yaml             # RTSP camera feed configuration
│   └── zones.yaml               # Virtual fence polygon definitions
│
├── docker-compose.yml           # Multi-container service orchestrator
├── .env                         # Environment variables template
└── README.md                    # System documentation
```

---

## ⚡ Quick Start

### Using Docker Compose
```bash
docker-compose up --build
```

### Manual Setup
1. **Backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload --port 8000
   ```

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```