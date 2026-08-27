# VISION — AI-Powered Border Surveillance & Video Analytics Platform

**VISION** is an AI-powered Intelligent Border Video Analytics Platform. This repository contains the core vision processing engine, supporting real-time video stream ingestion, object detection (YOLO11n), and multi-object tracking (ByteTrack).

---

## 📚 Project Documentation

All detailed specifications, PRDs, slice summaries, and architecture reports are organized in the [`docs/`](file:///c:/Career/Hackathons/SIH/VISION/docs) directory:

- 📄 **[PROJECT_SUMMARY.md](file:///c:/Career/Hackathons/SIH/VISION/docs/PROJECT_SUMMARY.md)**: Overall status report and architecture overview.
- 📄 **[VISION_PRD.md](file:///c:/Career/Hackathons/SIH/VISION/docs/VISION_PRD.md)**: Product Requirements Document.
- 📄 **[SIH_border_surveillance.md](file:///c:/Career/Hackathons/SIH/VISION/docs/SIH_border_surveillance.md)**: System design and technology selection guide.
- 📄 **[SLICE_1_SUMMARY.md](file:///c:/Career/Hackathons/SIH/VISION/docs/SLICE_1_SUMMARY.md)**: Vertical Slice 1 implementation & verification summary.

---

## 🔄 Current Pipeline (Vertical Slice 2)

```text
MP4 Video
    ↓
OpenCV VideoSource (src/ingestion/video.py)
    ↓
YOLO11n Detection (src/detection/detector.py)
    ↓
ByteTrack Tracking (src/tracking/tracker.py)
    ↓
Persistent Track Bounding Boxes & HUD (src/main.py)
    ↓
Real-Time OpenCV Display
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

```bash
python -m src.main --video data/videos/sample.mp4
```

### Command Line Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--video` | Path to input MP4 video file | `data/videos/test.mp4` |
| `--confidence` | Confidence threshold for object detection | `0.25` |
| `--interval` | Run YOLO inference every Nth frame | `1` |
| `--model` | YOLO model name or checkpoint path | `yolo11n.pt` |

---

## 🎮 Controls

* **`q`**: Quit application cleanly and display terminal tracking summary.