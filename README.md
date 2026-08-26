# VISION — Vertical Slice 1

**VISION** is an AI-powered Intelligent Border Video Analytics Platform. This repository contains **Vertical Slice 1**, which proves video ingestion and real-time object detection on conventional CCTV footage.

---

## 🔄 Current Pipeline

```text
MP4 Video
    ↓
OpenCV VideoSource
    ↓
Frame Extraction
    ↓
YOLO11n Detection
    ↓
Surveillance Target Filtering
    ↓
Bounding Box & HUD Annotation
    ↓
Real-Time OpenCV Display
```

---

## 🛠 Python Environment Setup

### 1. Create the Virtual Environment

Run the following command in the project root to create a local virtual environment in `.venv/`:

```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment

Choose the activation command for your operating system and shell:

* **Windows PowerShell:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

* **Windows Command Prompt (CMD):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

* **macOS / Linux (Bash or Zsh):**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Requirements

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running Vertical Slice 1

Place a sample MP4 video file at `data/videos/test.mp4` (or pass custom `--video` path):

```bash
python -m src.main --video data/videos/test.mp4
```

### Command Line Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--video` | Path to input MP4 video file | `data/videos/test.mp4` |
| `--confidence` | Confidence threshold for object detection | `0.25` |
| `--interval` | Run YOLO inference every Nth frame | `1` |
| `--model` | YOLO model name or checkpoint path | `yolo11n.pt` |

#### Custom Example:
```bash
python -m src.main --video data/videos/border_camera_01.mp4 --confidence 0.30 --interval 2
```

---

## 🎮 Controls

* **`q`**: Quit application cleanly and close display windows.

---

## 🛑 Current Limitations (Slice 1 Scope)

This vertical slice is intentionally limited to video ingestion, object detection, and visualization. It does **NOT** yet include:

* Object tracking (ByteTrack)
* Face detection (SCRFD) & recognition (ArcFace)
* Automatic Number Plate Recognition (ANPR) / OCR
* Virtual fence intrusion detection & events
* Storage (PostgreSQL, Redis, pgvector)
* Backend API (FastAPI, WebSockets)
* Frontend Dashboard (React)