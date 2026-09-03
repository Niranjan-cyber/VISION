// Mirrors the JSON contract produced by src/pipeline/serialize.py.
// Keep in sync with that file — this is the single source of truth for
// what the backend actually sends, not an aspirational shape.

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
  /** Present only when ANPR is enabled for this session — never a fake value. */
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

export interface SystemStatus {
  video: boolean;
  detection: boolean;
  tracking: boolean;
  face_id: boolean;
  anpr: boolean;
  events: boolean;
  error?: string | null;
}

export interface DetectionState {
  timestamp: number;
  frame_id: number;
  persons: Person[];
  vehicles: Vehicle[];
  statistics: Statistics;
  anpr_enabled: boolean;
  status: SystemStatus;
}

export type EventType =
  | "INTRUSION"
  | "UNKNOWN_PERSON_INTRUSION"
  | "LOITERING"
  | "SUSPICIOUS_VEHICLE";

export type Severity = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export interface AlertItem {
  alert_id: string;
  event_id: string;
  severity: Severity;
  title: string;
  message: string;
  status: "NEW" | "ACKNOWLEDGED" | "RESOLVED";
  timestamp: number;
  zone_name: string | null;
  object_type: string | null;
  track_id: number | null;
}
