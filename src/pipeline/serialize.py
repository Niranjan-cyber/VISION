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


def serialize_alert(alr: Alert, track_id: Optional[int] = None) -> Dict[str, Any]:
    return {
        "alert_id": alr.alert_id,
        "event_id": alr.event_id,
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


def serialize_events(session: PipelineSession, limit: int = 20) -> List[Dict[str, Any]]:
    """Builds the recent-alerts payload (GET /events), newest first."""
    if session.event_engine is None:
        return []
    # Alert has no direct track_id field (only SecurityEvent does) — look it
    # up via the shared event_id rather than duplicating tracking logic.
    track_id_by_event_id = {evt.event_id: evt.track_id for evt in session.event_engine.event_history}
    recent = session.event_engine.active_alerts[-limit:]
    return [
        serialize_alert(a, track_id=track_id_by_event_id.get(a.event_id))
        for a in reversed(recent)
    ]
