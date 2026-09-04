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
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.core.device import resolve_yolo_device
from src.events.types import AlertStatus
from src.ingestion.video import discover_camera_devices
from src.pipeline.camera_manager import AI_FPS_DEFAULT, CameraLimitReached, CameraManager, MAX_ACTIVE_CAMERAS
from src.pipeline.serialize import (
    serialize_camera_summary,
    serialize_events,
    serialize_global_detections,
    serialize_global_status,
    serialize_state,
)
from src.storage import InvalidAlertTransition, StoredAlert, StoredEvent, StoredZone

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


# NOTE: GET /events used to return the live in-memory alert feed (Phase 2).
# Phase 3 splits that cleanly in two: GET /alerts is the operator-facing,
# persistent, lifecycle-managed alert feed (below); GET /events is now
# historical event search over the same persistent store. serialize_events/
# serialize_global_events (src/pipeline/serialize.py) are unchanged and
# still used by GET /cameras/{id}/events, which keeps its original
# in-memory-per-camera contract.


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


# ---------------------------------------------------------------------------
# Phase 3 — serialization helpers (DB rows -> API JSON)
# ---------------------------------------------------------------------------
def _serialize_event(event: StoredEvent) -> Dict[str, Any]:
    return {
        "event_id": event.id,
        "camera_id": event.camera_id,
        "camera_name": event.camera_name,
        "source_type": event.source_type,
        "event_type": event.event_type,
        "severity": event.severity,
        "timestamp": event.timestamp,
        "created_at": event.created_at,
        "track_id": event.track_id,
        "identity": event.identity,
        "zone_id": event.zone_id,
        "zone_name": event.zone_name,
        "description": event.description,
        "metadata": event.metadata,
        "has_snapshot": bool(event.snapshot_path) and os.path.exists(event.snapshot_path),
    }


def _serialize_alert(alert: StoredAlert, event: Optional[StoredEvent]) -> Dict[str, Any]:
    """Alerts and events are separate rows (see docs/PHASE3.md) — this joins
    them for display so the API keeps returning the zone/track/identity
    fields the frontend already expects from an alert, without duplicating
    that data into the alerts table itself."""
    return {
        "alert_id": alert.id,
        "event_id": alert.event_id,
        "camera_id": alert.camera_id,
        "camera_name": alert.camera_name,
        "event_type": event.event_type if event else None,
        "severity": alert.severity,
        "title": alert.title,
        "message": alert.message,
        "status": alert.status,
        "timestamp": event.timestamp if event else None,
        "created_at": alert.created_at,
        "acknowledged_at": alert.acknowledged_at,
        "resolved_at": alert.resolved_at,
        "zone_id": event.zone_id if event else None,
        "zone_name": event.zone_name if event else None,
        "track_id": event.track_id if event else None,
        "identity": event.identity if event else None,
        "object_type": (event.metadata.get("object_type") if event else None),
    }


def _serialize_zone(zone: StoredZone) -> Dict[str, Any]:
    return {
        "id": zone.id,
        "camera_id": zone.camera_id,
        "name": zone.name,
        "type": zone.type,
        "polygon": [list(p) for p in zone.polygon],
        "enabled": zone.enabled,
        "created_at": zone.created_at,
        "updated_at": zone.updated_at,
    }


def _sync_in_memory_alert_status(alert: StoredAlert, target_status: str) -> None:
    """Acknowledge/resolve must also be reflected on the live camera's
    in-memory EventEngine.active_alerts (which GET /cameras/{id}/events
    still reads — see the note above) — otherwise an operator who
    acknowledges via the global Alerts page would see a stale NEW badge on
    the camera's own focus view. Best-effort: the persistent DB row (the
    real source of truth) is already updated by this point regardless."""
    cam = camera_manager.get(alert.camera_id)
    if cam is None:
        return
    session, _ = cam.snapshot()
    if session is None or session.event_engine is None:
        return
    for a in session.event_engine.active_alerts:
        if a.alert_id == alert.id:
            a.status = AlertStatus(target_status)
            break


# ---------------------------------------------------------------------------
# Phase 3 — Alert management (persistent, filterable, lifecycle-managed)
# ---------------------------------------------------------------------------
@app.get("/alerts")
def list_alerts(
    camera_id: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    event_type: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    alerts = camera_manager.alert_repo.query(
        camera_id=camera_id, severity=severity, status=status, event_type=event_type,
        start_time=start_time, end_time=end_time, limit=limit, offset=offset,
    )
    return [_serialize_alert(a, camera_manager.event_repo.get(a.event_id)) for a in alerts]


@app.get("/alerts/{alert_id}")
def get_alert(alert_id: str):
    alert = camera_manager.alert_repo.get(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"No alert with id '{alert_id}'")
    return _serialize_alert(alert, camera_manager.event_repo.get(alert.event_id))


@app.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str):
    try:
        alert = camera_manager.alert_repo.transition(alert_id, "ACKNOWLEDGED")
    except InvalidAlertTransition as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    _sync_in_memory_alert_status(alert, "ACKNOWLEDGED")
    return _serialize_alert(alert, camera_manager.event_repo.get(alert.event_id))


@app.post("/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: str):
    try:
        alert = camera_manager.alert_repo.transition(alert_id, "RESOLVED")
    except InvalidAlertTransition as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    _sync_in_memory_alert_status(alert, "RESOLVED")
    return _serialize_alert(alert, camera_manager.event_repo.get(alert.event_id))


# ---------------------------------------------------------------------------
# Phase 3 — Historical event search
# ---------------------------------------------------------------------------
@app.get("/events")
def list_events(
    camera_id: Optional[str] = None,
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    identity: Optional[str] = None,
    track_id: Optional[int] = None,
    status: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    events = camera_manager.event_repo.query(
        camera_id=camera_id, event_type=event_type, severity=severity, identity=identity,
        track_id=track_id, status=status, start_time=start_time, end_time=end_time,
        limit=limit, offset=offset,
    )
    return [_serialize_event(e) for e in events]


@app.get("/events/{event_id}")
def get_event(event_id: str):
    event = camera_manager.event_repo.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"No event with id '{event_id}'")
    return _serialize_event(event)


@app.get("/events/{event_id}/snapshot")
def get_event_snapshot(event_id: str):
    event = camera_manager.event_repo.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"No event with id '{event_id}'")
    if not event.snapshot_path or not os.path.exists(event.snapshot_path):
        raise HTTPException(status_code=404, detail="No snapshot available for this event")
    return FileResponse(event.snapshot_path, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Phase 3 — Incident investigation
# ---------------------------------------------------------------------------
@app.get("/investigations/event/{event_id}")
def investigate_event(event_id: str):
    event = camera_manager.event_repo.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"No event with id '{event_id}'")

    alert = camera_manager.alert_repo.get_by_event(event_id)
    related: List[StoredEvent] = []
    if event.track_id is not None:
        related = camera_manager.event_repo.related_to(event.camera_id, event.track_id, exclude_event_id=event_id)

    return {
        "event": _serialize_event(event),
        "alert": _serialize_alert(alert, event) if alert else None,
        "related_events": [_serialize_event(e) for e in related],
    }


@app.get("/investigations/person/{identity}")
def investigate_person(identity: str):
    """Aggregates a *recognized* identity's events across every camera it
    was seen on. 'UNKNOWN' is not a real identity — see
    /investigations/track for an unrecognized person or a vehicle, the only
    honest identifier the pipeline has for either."""
    if identity.strip().upper() == "UNKNOWN":
        raise HTTPException(
            status_code=400,
            detail="'UNKNOWN' is not a specific identity — use /investigations/track/{camera_id}/{track_id} instead",
        )
    events = camera_manager.event_repo.for_identity(identity, limit=200)
    if not events:
        raise HTTPException(status_code=404, detail=f"No events found for identity '{identity}'")
    cameras_seen = sorted({e.camera_id for e in events})
    last_seen = events[0].created_at  # for_identity() orders newest first
    return {
        "identity": identity,
        "recognized": True,
        "cameras": cameras_seen,
        "last_seen": last_seen,
        "events": [_serialize_event(e) for e in events],
    }


@app.get("/investigations/track/{camera_id}/{track_id}")
def investigate_track(camera_id: str, track_id: int):
    """Investigation for a single (camera, track) — the identifier the
    pipeline actually has for an unrecognized person or any vehicle. Never
    claims cross-camera identity for an entity the system can't recognize."""
    events = camera_manager.event_repo.related_to(camera_id, track_id, limit=200)
    if not events:
        raise HTTPException(status_code=404, detail=f"No events found for camera '{camera_id}' track #{track_id}")
    latest = events[-1]  # related_to() orders oldest first
    return {
        "camera_id": camera_id,
        "track_id": track_id,
        "object_type": latest.metadata.get("object_type"),
        "identity": latest.identity,
        "last_seen": latest.created_at,
        "plate": latest.metadata.get("plate"),
        "plate_confidence": latest.metadata.get("plate_confidence"),
        "events": [_serialize_event(e) for e in events],
    }


# ---------------------------------------------------------------------------
# Phase 3 — Zone management
# ---------------------------------------------------------------------------
class ZoneCreateRequest(BaseModel):
    camera_id: str
    name: str
    type: str = "restricted"
    polygon: List[List[int]]
    enabled: bool = True


class ZoneUpdateRequest(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    polygon: Optional[List[List[int]]] = None
    enabled: Optional[bool] = None


@app.get("/zones")
def list_zones(camera_id: Optional[str] = None):
    if camera_id:
        zones = camera_manager.zone_repo.list_for_camera(camera_id)
    else:
        zones = []
        for cam in camera_manager.list_cameras():
            zones.extend(camera_manager.zone_repo.list_for_camera(cam.camera_id))
    return [_serialize_zone(z) for z in zones]


@app.get("/zones/{zone_id}")
def get_zone(zone_id: str):
    zone = camera_manager.zone_repo.get(zone_id)
    if zone is None:
        raise HTTPException(status_code=404, detail=f"No zone with id '{zone_id}'")
    return _serialize_zone(zone)


@app.post("/zones")
def create_zone(body: ZoneCreateRequest):
    _camera_or_404(body.camera_id)
    zone_id = f"zone_{uuid.uuid4().hex[:10]}"
    try:
        zone = camera_manager.zone_repo.create(zone_id, body.camera_id, body.name, body.type, body.polygon, body.enabled)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    camera_manager.refresh_camera_zones(body.camera_id)
    return _serialize_zone(zone)


@app.put("/zones/{zone_id}")
def update_zone(zone_id: str, body: ZoneUpdateRequest):
    existing = camera_manager.zone_repo.get(zone_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"No zone with id '{zone_id}'")
    try:
        zone = camera_manager.zone_repo.update(
            zone_id, name=body.name, zone_type=body.type, polygon=body.polygon, enabled=body.enabled,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    camera_manager.refresh_camera_zones(existing.camera_id)
    return _serialize_zone(zone)


@app.delete("/zones/{zone_id}")
def delete_zone(zone_id: str):
    existing = camera_manager.zone_repo.get(zone_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"No zone with id '{zone_id}'")
    camera_manager.zone_repo.delete(zone_id)
    camera_manager.refresh_camera_zones(existing.camera_id)
    return {"deleted": zone_id}


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
