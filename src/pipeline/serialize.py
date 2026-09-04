"""
JSON-serialization helpers that turn a PipelineSession's current state into
the normalized API contract the React dashboard consumes. Reuses the existing
ObjectState / SecurityEvent / Alert dataclasses verbatim — no parallel data
model, no re-derivation of anything the AI pipeline already computed.
"""
from typing import Any, Dict, List, Optional

from src.core.types import BoundingBox
from src.events.state import ObjectState
from src.events.types import Alert, SecurityEvent
from src.pipeline.session import FrameResult, PipelineSession

TARGET_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


def _bbox_list(bbox: BoundingBox) -> List[int]:
    return [bbox.x1, bbox.y1, bbox.x2, bbox.y2]


def serialize_person(obj: ObjectState) -> Dict[str, Any]:
    """
    identity is one of:
      - None        -> no face detected on this track (NOT the same as unknown)
      - "UNKNOWN"    -> face detected, did not match the gallery
      - a name       -> face detected and matched (see face_similarity for the score)
    """
    return {
        "track_id": obj.track_id,
        "identity": obj.identity,
        "face_similarity": round(obj.face_similarity, 3) if obj.face_similarity is not None else None,
        "bbox": _bbox_list(obj.bbox),
        "confidence": round(obj.confidence, 3),
        "zone": obj.current_zone,
    }


def serialize_vehicle(obj: ObjectState, anpr_enabled: bool) -> Dict[str, Any]:
    """Plate fields are only included when ANPR is actually enabled for this
    session — never a placeholder/fake value when it's off."""
    v: Dict[str, Any] = {
        "track_id": obj.track_id,
        "type": obj.object_type,
        "bbox": _bbox_list(obj.bbox),
        "confidence": round(obj.confidence, 3),
        "zone": obj.current_zone,
    }
    if anpr_enabled:
        v["plate"] = obj.plate
        v["plate_confidence"] = round(obj.plate_confidence, 3) if obj.plate_confidence is not None else None
    return v


def serialize_event(evt: SecurityEvent) -> Dict[str, Any]:
    return {
        "event_id": evt.event_id,
        "event_type": evt.event_type.value,
        "severity": evt.severity.value,
        "track_id": evt.track_id,
        "zone_id": evt.zone_id,
        "zone_name": evt.metadata.get("zone_name"),
        "message": evt.message,
        "timestamp": evt.timestamp,
    }


def serialize_alert(
    alr: Alert,
    camera_id: str,
    camera_name: str,
    track_id: Optional[int] = None,
) -> Dict[str, Any]:
    """camera_id/camera_name are required — every alert must identify which
    camera raised it (Phase 2 requirement: events belong to a camera)."""
    return {
        "alert_id": alr.alert_id,
        "event_id": alr.event_id,
        "camera_id": camera_id,
        "camera_name": camera_name,
        "severity": alr.severity.value,
        "title": alr.title,
        "message": alr.message,
        "status": alr.status.value,
        "timestamp": alr.timestamp,
        "zone_name": alr.metadata.get("zone_name"),
        "object_type": alr.metadata.get("object_type"),
        "track_id": track_id,
    }


def serialize_state(session: PipelineSession, result: Optional[FrameResult]) -> Dict[str, Any]:
    """Builds the full normalized dashboard payload (GET /detections)."""
    object_states = session.latest_object_states

    persons = [serialize_person(o) for o in object_states if o.object_type == "person"]
    vehicles = [
        serialize_vehicle(o, session.enable_anpr)
        for o in object_states
        if o.object_type in TARGET_VEHICLE_CLASSES
    ]

    recognized = sum(1 for m in session.track_identity_cache.values() if m.is_match)
    active_events = len(session.event_engine.event_history) if session.event_engine else 0

    return {
        "timestamp": result.timestamp if result else 0.0,
        "frame_id": result.frame_index if result else session.frame_index,
        "persons": persons,
        "vehicles": vehicles,
        "statistics": {
            "persons": len(persons),
            "vehicles": len(vehicles),
            "faces_detected": len(session.latest_faces),
            "recognized_faces": recognized,
            "active_events": active_events,
        },
        "anpr_enabled": session.enable_anpr,
        "status": session.status,
    }


def serialize_events(
    session: PipelineSession,
    camera_id: str,
    camera_name: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Builds one camera's recent-alerts payload (GET /cameras/{id}/events), newest first."""
    if session.event_engine is None:
        return []
    # Alert has no direct track_id field (only SecurityEvent does) — look it
    # up via the shared event_id rather than duplicating tracking logic.
    track_id_by_event_id = {evt.event_id: evt.track_id for evt in session.event_engine.event_history}
    recent = session.event_engine.active_alerts[-limit:]
    return [
        serialize_alert(a, camera_id, camera_name, track_id=track_id_by_event_id.get(a.event_id))
        for a in reversed(recent)
    ]


def serialize_camera_summary(camera) -> Dict[str, Any]:
    """Builds one entry of the GET /cameras list. `camera` is a CameraSession —
    typed loosely here to avoid a circular import with camera_manager."""
    session, _ = camera.snapshot()
    cfg = camera.config
    display_source = (
        f"Camera {cfg.device_index} (live)" if cfg.source_type == "live" else cfg.video_path
    )
    summary: Dict[str, Any] = {
        "camera_id": camera.camera_id,
        "camera_name": camera.camera_name,
        "source_type": cfg.source_type,
        "video_source": display_source,
        "device_index": cfg.device_index,
        "zones_path": cfg.zones_path,
        "has_zone": cfg.zones_path is not None,
        "status": camera.status,
        "error": camera.error,
    }
    if session is not None:
        summary["anpr_enabled"] = session.enable_anpr
        recognized = sum(1 for m in session.track_identity_cache.values() if m.is_match)
        active_events = len(session.event_engine.event_history) if session.event_engine else 0
        object_states = session.latest_object_states
        persons = sum(1 for o in object_states if o.object_type == "person")
        vehicles = sum(1 for o in object_states if o.object_type in TARGET_VEHICLE_CLASSES)
        summary["statistics"] = {
            "persons": persons,
            "vehicles": vehicles,
            "faces_detected": len(session.latest_faces),
            "recognized_faces": recognized,
            "active_events": active_events,
        }
    else:
        summary["anpr_enabled"] = camera.config.enable_anpr
        summary["statistics"] = {
            "persons": 0, "vehicles": 0, "faces_detected": 0,
            "recognized_faces": 0, "active_events": 0,
        }
    return summary


def serialize_global_status(manager) -> Dict[str, Any]:
    """GET /status — aggregate health across every configured camera."""
    from src.pipeline.camera_manager import MAX_ACTIVE_CAMERAS

    cameras = manager.list_cameras()
    return {
        "cameras_active": len(cameras),
        "cameras_max": MAX_ACTIVE_CAMERAS,
        "cameras": {
            cam.camera_id: {
                "camera_name": cam.camera_name,
                "status": cam.status,
                "error": cam.error,
                "source_type": cam.config.source_type,
            }
            for cam in cameras
        },
    }


def serialize_global_detections(manager) -> Dict[str, Any]:
    """GET /detections — per-camera breakdown plus an aggregate total across
    every active camera. Never invents a metric that isn't a sum of real
    per-camera fields."""
    cameras_payload = []
    totals = {"persons": 0, "vehicles": 0, "faces_detected": 0, "recognized_faces": 0, "active_events": 0}

    for cam in manager.list_cameras():
        session, result = cam.snapshot()
        cam_state = serialize_state(session, result) if session is not None else {
            "timestamp": 0.0, "frame_id": 0, "persons": [], "vehicles": [],
            "statistics": {"persons": 0, "vehicles": 0, "faces_detected": 0, "recognized_faces": 0, "active_events": 0},
            "anpr_enabled": cam.config.enable_anpr, "status": {},
        }
        cameras_payload.append({
            "camera_id": cam.camera_id,
            "camera_name": cam.camera_name,
            "camera_status": cam.status,
            "source_type": cam.config.source_type,
            "has_zone": cam.config.zones_path is not None,
            **cam_state,
        })
        for k in totals:
            totals[k] += cam_state["statistics"][k]

    totals["cameras_active"] = len(cameras_payload)
    return {"cameras": cameras_payload, "statistics": totals}


def serialize_global_events(manager, per_camera_limit: int = 20, total_limit: int = 40) -> List[Dict[str, Any]]:
    """GET /events — every camera's recent alerts merged, newest first."""
    merged: List[Dict[str, Any]] = []
    for cam in manager.list_cameras():
        session, _ = cam.snapshot()
        if session is None:
            continue
        merged.extend(serialize_events(session, cam.camera_id, cam.camera_name, limit=per_camera_limit))
    merged.sort(key=lambda a: a["timestamp"], reverse=True)
    return merged[:total_limit]
