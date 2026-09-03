// Mirrors the JSON contract produced by src/pipeline/serialize.py and
// backend/main.py. Keep in sync with those files — this is the single
// source of truth for what the backend actually sends, not an aspirational
// shape. Multi-camera (Phase 2): every camera-scoped value now carries
// camera_id/camera_name; global endpoints aggregate across active cameras.

export type CameraStatus = "starting" | "online" | "stopped" | "error";

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

/** One security alert (GET /events global, GET /cameras/{id}/events). */
export interface AlertItem {
  alert_id: string;
  event_id: string;
  camera_id: string;
  camera_name: string;
  severity: Severity;
  title: string;
  message: string;
  status: "NEW" | "ACKNOWLEDGED" | "RESOLVED";
  timestamp: number;
  zone_name: string | null;
  object_type: string | null;
  track_id: number | null;
}

/** GET /status (global aggregate). */
export interface GlobalStatus {
  cameras_active: number;
  cameras_max: number;
  cameras: Record<string, { camera_name: string; status: CameraStatus; error: string | null }>;
}

/** One entry of GET /cameras. */
export interface CameraSummary {
  camera_id: string;
  camera_name: string;
  video_source: string;
  zones_path: string | null;
  status: CameraStatus;
  error: string | null;
  anpr_enabled: boolean;
  statistics: Statistics;
}
