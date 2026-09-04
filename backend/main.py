"""
VISION FastAPI backend — multi-camera command center.

Thin API layer between the existing Python AI pipeline
(src/pipeline/session.py, src/pipeline/camera_manager.py) and the React
dashboard. This module contains NO detection/tracking/face/ANPR/event logic
of its own — it only:
  1. Owns one CameraManager, which runs up to MAX_ACTIVE_CAMERAS independent
     CameraSession objects (each one PipelineSession in its own thread).
  2. Publishes per-camera and aggregate state as JSON, and per-camera
     annotated frames as MJPEG.
  3. Accepts new video uploads or live local camera devices and turns them
     into new camera sessions.

No Redis/Kafka/database — CameraManager's in-memory dict of CameraSession
objects is the entire "multi-camera infrastructure" this MVP needs.
"""
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.core.device import resolve_yolo_device
from src.ingestion.video import discover_camera_devices
from src.pipeline.camera_manager import AI_FPS_DEFAULT, CameraLimitReached, CameraManager, MAX_ACTIVE_CAMERAS
from src.pipeline.serialize import (
    serialize_camera_summary,
    serialize_events,
    serialize_global_detections,
    serialize_global_events,
    serialize_global_status,
    serialize_state,
)

# ---------------------------------------------------------------------------
# Golden demo configuration.
# CAM-01 is the original single-camera golden demo (unchanged video/zone) —
# see configs/zones_demo.yaml for calibration notes. CAM-02/03/04 use the
# next-best verified videos in data/videos/, each with its own calibrated
# zone file (configs/zones_cam0{2,3,4}.yaml) — no invented footage.
# ---------------------------------------------------------------------------
DEFAULT_CAMERAS = [
    {
        "camera_name": os.environ.get("VISION_CAM01_NAME", "Border Gate"),
        "video_path": os.environ.get("VISION_DEMO_VIDEO", "data/videos/shreyas1.mp4"),
        "zones_path": os.environ.get("VISION_DEMO_ZONES", "configs/zones_demo.yaml"),
        "loitering_duration": float(os.environ.get("VISION_LOITERING_DURATION", "3.0")),
    },
    {
        "camera_name": "BOP East",
        "video_path": "data/videos/jaysingpure1.mp4",
        "zones_path": "configs/zones_cam02.yaml",
        "loitering_duration": 3.0,
    },
    {
        "camera_name": "Perimeter Road",
        "video_path": "data/videos/sample1.mp4",
        "zones_path": "configs/zones_cam03.yaml",
        "loitering_duration": 4.0,
    },
    {
        "camera_name": "Restricted Zone",
        "video_path": "data/videos/salman4.mp4",
        "zones_path": "configs/zones_cam04.yaml",
        "loitering_duration": 3.0,
    },
]
# Lets tests / a lean single-camera run boot fast without waiting on 4x model loads.
CAMERA_COUNT = int(os.environ.get("VISION_CAMERA_COUNT", "4"))
# ANPR defaults OFF everywhere: no shipped video has a verified-legible plate,
# and easyocr isn't installed — see docs/README ANPR decision notes.
ENABLE_ANPR = os.environ.get("VISION_ENABLE_ANPR", "false").strip().lower() == "true"
# 'auto' (default) uses CUDA if torch/onnxruntime actually confirm it's available,
# else CPU. 'cuda' forces the attempt (still falls back to CPU with a warning if
# unavailable — see src/core/device.py). 'cpu' forces CPU regardless of hardware.
DEVICE = os.environ.get("VISION_DEVICE", "auto").strip().lower()
# Rate the AI worker (detection/tracking/face/ANPR/events) runs at, decoupled
# from video capture/display FPS — see src/pipeline/camera_manager.py.
AI_FPS = float(os.environ.get("VISION_AI_FPS", str(AI_FPS_DEFAULT)))
# How many local device indices GET /cameras/devices probes for a live camera.
CAMERA_DEVICE_PROBE_RANGE = int(os.environ.get("VISION_CAMERA_DEVICE_PROBE_RANGE", "5"))

UPLOAD_DIR = "data/uploads"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
JPEG_STREAM_POLL_INTERVAL_S = 0.05

camera_manager = CameraManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    for cfg in DEFAULT_CAMERAS[:CAMERA_COUNT]:
        try:
            camera_manager.add_camera(enable_anpr=ENABLE_ANPR, device=DEVICE, ai_fps=AI_FPS, **cfg)
        except CameraLimitReached:
            break
    yield
    camera_manager.shutdown()


app = FastAPI(title="VISION Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP demo only, single trusted machine — see README
    allow_methods=["*"],
    allow_headers=["*"],
)


def _camera_or_404(camera_id: str):
    cam = camera_manager.get(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"No camera with id '{camera_id}'")
    return cam


def _ai_engine_status() -> dict:
    """Reports which device each pipeline stage is actually running on.
    Prefers the *actually-granted* provider read off a live camera's session
    (never just what was requested — see src/core/device.py's discipline of
    checking session.get_providers() after the fact) and only falls back to
    the resolved-but-unconfirmed device before any camera has come online."""
    yolo_device = "CUDA" if resolve_yolo_device(DEVICE).startswith("cuda") else "CPU"
    face_device = None
    for cam in camera_manager.list_cameras():
        session, _ = cam.snapshot()
        if session is None:
            continue
        yolo_device = "CUDA" if session.detector.device.startswith("cuda") else "CPU"
        provider = getattr(session.face_embedder, "active_provider", "CPUExecutionProvider")
        face_device = "CUDA" if provider == "CUDAExecutionProvider" else "CPU"
        break
    if face_device is None:
        # No camera online yet — report the resolved device without
        # claiming a specific onnxruntime provider was actually granted.
        face_device = yolo_device
    return {
        "yolo_device": yolo_device,
        "face_recognition_device": face_device,
        "yunet_device": "CPU",
        "tracking_device": "CPU",
        "event_engine_device": "CPU",
        "ai_fps": AI_FPS,
    }


# ---------------------------------------------------------------------------
# Global (aggregate) endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def global_status():
    payload = serialize_global_status(camera_manager)
    payload["ai_engine"] = _ai_engine_status()
    return payload


@app.get("/detections")
def global_detections():
    return serialize_global_detections(camera_manager)


@app.get("/events")
def global_events():
    return serialize_global_events(camera_manager)


# ---------------------------------------------------------------------------
# Camera management
# ---------------------------------------------------------------------------
@app.get("/cameras")
def list_cameras():
    return [serialize_camera_summary(cam) for cam in camera_manager.list_cameras()]


@app.get("/cameras/devices")
def list_camera_devices():
    """Probes local camera device indices by actually attempting to open
    (and read a frame from) each one — never invents a device that can't
    actually be opened. A device already claimed by an active live
    CameraSession will correctly report unavailable (it's in use)."""
    return discover_camera_devices(max_index=CAMERA_DEVICE_PROBE_RANGE)


@app.post("/cameras")
def add_camera(
    camera_name: str = Form(...),
    source_type: str = Form("video"),
    video: Optional[UploadFile] = File(None),
    device_index: Optional[int] = Form(None),
):
    source_type = source_type.strip().lower()
    if source_type not in ("video", "live"):
        raise HTTPException(status_code=400, detail="source_type must be 'video' or 'live'")

    name = camera_name.strip() or "Untitled Camera"

    if source_type == "live":
        if device_index is None:
            raise HTTPException(status_code=400, detail="device_index is required for source_type='live'")
        try:
            cam = camera_manager.add_camera(
                camera_name=name,
                source_type="live",
                device_index=device_index,
                zones_path=None,  # no auto-calibrated zone for a newly connected camera
                enable_anpr=ENABLE_ANPR,
                device=DEVICE,
                ai_fps=AI_FPS,
            )
        except CameraLimitReached as e:
            raise HTTPException(status_code=409, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return serialize_camera_summary(cam)

    # source_type == "video": retain the existing upload implementation —
    # extension validation, a safe generated filename, MAX_ACTIVE_CAMERAS.
    if video is None:
        raise HTTPException(status_code=400, detail="video file is required for source_type='video'")

    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}",
        )

    # Never trust the original filename as a path/identifier.
    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = os.path.join(UPLOAD_DIR, stored_name)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(stored_path, "wb") as f:
        f.write(video.file.read())

    try:
        cam = camera_manager.add_camera(
            camera_name=name,
            source_type="video",
            video_path=stored_path,
            zones_path=None,  # no auto-calibrated zone for an arbitrary upload
            enable_anpr=ENABLE_ANPR,
            device=DEVICE,
            ai_fps=AI_FPS,
        )
    except CameraLimitReached as e:
        os.remove(stored_path)
        raise HTTPException(status_code=409, detail=str(e))

    return serialize_camera_summary(cam)


@app.delete("/cameras/{camera_id}")
def remove_camera(camera_id: str):
    if not camera_manager.remove_camera(camera_id):
        raise HTTPException(status_code=404, detail=f"No camera with id '{camera_id}'")
    return {"removed": camera_id}


@app.post("/cameras/{camera_id}/restart")
def restart_camera(camera_id: str):
    if not camera_manager.restart_camera(camera_id):
        raise HTTPException(status_code=404, detail=f"No camera with id '{camera_id}'")
    return {"restarting": camera_id}


# ---------------------------------------------------------------------------
# Per-camera endpoints
# ---------------------------------------------------------------------------
@app.get("/cameras/{camera_id}/status")
def camera_status(camera_id: str):
    cam = _camera_or_404(camera_id)
    return {
        "camera_id": cam.camera_id,
        "camera_name": cam.camera_name,
        "status": cam.status,
        "error": cam.error,
    }


@app.get("/cameras/{camera_id}/detections")
def camera_detections(camera_id: str):
    cam = _camera_or_404(camera_id)
    session, result = cam.snapshot()
    if session is None:
        return JSONResponse(status_code=503, content={"error": cam.error or "camera is still starting"})
    return serialize_state(session, result)


@app.get("/cameras/{camera_id}/events")
def camera_events(camera_id: str):
    cam = _camera_or_404(camera_id)
    session, _ = cam.snapshot()
    if session is None:
        return JSONResponse(status_code=503, content={"error": cam.error or "camera is still starting"})
    return serialize_events(session, cam.camera_id, cam.camera_name)


def _mjpeg_generator(cam):
    boundary = b"--frame"
    while True:
        jpeg = cam.latest_jpeg()
        if jpeg is not None:
            yield (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg)).encode()
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
        time.sleep(JPEG_STREAM_POLL_INTERVAL_S)


@app.get("/cameras/{camera_id}/stream")
def camera_stream(camera_id: str):
    cam = _camera_or_404(camera_id)
    return StreamingResponse(
        _mjpeg_generator(cam),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
