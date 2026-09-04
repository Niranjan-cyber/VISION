// Mirrors the JSON contract produced by src/pipeline/serialize.py and
// backend/main.py. Keep in sync with those files — this is the single
// source of truth for what the backend actually sends, not an aspirational
// shape. Multi-camera (Phase 2): every camera-scoped value now carries
// camera_id/camera_name; global endpoints aggregate across active cameras.

export type CameraStatus = "starting" | "online" | "stopped" | "error";
export type SourceType = "video" | "live";

export interface Person {
  track_id: number;
  /** null = no face detected on this track. "UNKNOWN" = face detected but
   *  did not match the gallery. Any other string = a matched identity. */
  identity: string | null;
  face_similarity: number | null;
  bbox: [number, number, number, number];
  confidence: number;
  zone: string | null;
}

export interface Vehicle {
  track_id: number;
  type: string;
  bbox: [number, number, number, number];
  confidence: number;
  zone: string | null;
  /** Present only when ANPR is enabled for this camera — never a fake value. */
  plate?: string | null;
  plate_confidence?: number | null;
}

export interface Statistics {
  persons: number;
  vehicles: number;
  faces_detected: number;
  recognized_faces: number;
  active_events: number;
}

export interface SubsystemStatus {
  video: boolean;
  detection: boolean;
  tracking: boolean;
  face_id: boolean;
  anpr: boolean;
  events: boolean;
}

/** One camera's full detection payload (GET /cameras/{id}/detections, and
 * each entry of the global GET /detections .cameras[] array). */
export interface CameraDetectionState {
  camera_id: string;
  camera_name: string;
  camera_status: CameraStatus;
  source_type: SourceType;
  has_zone: boolean;
  timestamp: number;
  frame_id: number;
  persons: Person[];
  vehicles: Vehicle[];
  statistics: Statistics;
  anpr_enabled: boolean;
  status: SubsystemStatus;
}

/** GET /detections (global aggregate). */
export interface GlobalDetections {
  cameras: CameraDetectionState[];
  statistics: Statistics & { cameras_active: number };
}

export type Severity = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type AlertLifecycleStatus = "NEW" | "ACKNOWLEDGED" | "RESOLVED";

/** One operator alert (GET /alerts, GET /alerts/{id}, GET /cameras/{id}/events).
 * Persistent (SQLite) and lifecycle-managed as of Phase 3 — see docs/PHASE3.md. */
export interface AlertItem {
  alert_id: string;
  event_id: string;
  camera_id: string;
  camera_name: string;
  event_type: string | null;
  severity: Severity;
  title: string;
  message: string;
  status: AlertLifecycleStatus;
  /** Video-relative seconds (the clip's own clock), same convention as before. */
  timestamp: number | null;
  /** Wall-clock ISO8601 — survives a restart, unlike `timestamp`. */
  created_at: string;
  acknowledged_at: string | null;
  resolved_at: string | null;
  zone_id: string | null;
  zone_name: string | null;
  object_type: string | null;
  track_id: number | null;
  identity: string | null;
}

/** One historical event row (GET /events, GET /events/{id}). */
export interface EventItem {
  event_id: string;
  camera_id: string;
  camera_name: string;
  source_type: SourceType | null;
  event_type: string;
  severity: Severity;
  timestamp: number;
  created_at: string;
  track_id: number | null;
  identity: string | null;
  zone_id: string | null;
  zone_name: string | null;
  description: string;
  metadata: Record<string, unknown>;
  has_snapshot: boolean;
}

/** GET /investigations/event/{event_id}. */
export interface EventInvestigation {
  event: EventItem;
  alert: AlertItem | null;
  related_events: EventItem[];
}

/** GET /investigations/person/{identity}. */
export interface PersonInvestigation {
  identity: string;
  recognized: true;
  cameras: string[];
  last_seen: string;
  events: EventItem[];
}

/** GET /investigations/track/{camera_id}/{track_id} — an unrecognized
 * person or a vehicle, identified only by where the pipeline actually saw
 * it (never a fabricated cross-camera identity). */
export interface TrackInvestigation {
  camera_id: string;
  track_id: number;
  object_type: string | null;
  identity: string | null;
  last_seen: string;
  plate: string | null;
  plate_confidence: number | null;
  events: EventItem[];
}

export type ZoneType = "restricted" | "warning" | "monitored";

/** One surveillance zone (GET/POST/PUT/DELETE /zones). */
export interface ZoneItem {
  id: string;
  camera_id: string;
  name: string;
  type: ZoneType;
  polygon: [number, number][];
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

/** Which device each pipeline stage is actually running on, and the
 * configured AI-worker sampling rate — part of GET /status. */
export interface AIEngineStatus {
  yolo_device: "CUDA" | "CPU";
  face_recognition_device: "CUDA" | "CPU";
  yunet_device: "CPU";
  tracking_device: "CPU";
  event_engine_device: "CPU";
  ai_fps: number;
}

/** GET /status (global aggregate). */
export interface GlobalStatus {
  cameras_active: number;
  cameras_max: number;
  cameras: Record<string, { camera_name: string; status: CameraStatus; error: string | null; source_type: SourceType }>;
  ai_engine: AIEngineStatus;
}

/** One entry of GET /cameras. */
export interface CameraSummary {
  camera_id: string;
  camera_name: string;
  source_type: SourceType;
  video_source: string;
  device_index: number | null;
  zones_path: string | null;
  has_zone: boolean;
  status: CameraStatus;
  error: string | null;
  anpr_enabled: boolean;
  statistics: Statistics;
}

/** One entry of GET /cameras/devices — a probed local camera device index. */
export interface CameraDevice {
  device_index: number;
  available: boolean;
  width: number;
  height: number;
}
