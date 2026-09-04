import { useEffect, useRef, useState } from "react";
import type {
  AlertItem,
  CameraDetectionState,
  CameraDevice,
  CameraSummary,
  EventInvestigation,
  EventItem,
  PersonInvestigation,
  TrackInvestigation,
  ZoneItem,
  ZoneType,
  GlobalDetections,
  GlobalStatus,
} from "./types";

// Same-origin by default (works when the backend serves the build), override
// for local dev against a separately-running backend.
export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? body.error ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function sendJSON<T>(path: string, method: "POST" | "PUT" | "DELETE", body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const parsed = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(parsed.detail ?? `${res.status} ${res.statusText}`);
  }
  return parsed as T;
}

function toQuery(params: object): string {
  const usp = new URLSearchParams();
  for (const [k, v] of Object.entries(params as Record<string, string | number | undefined | null>)) {
    if (v !== undefined && v !== null && v !== "") usp.set(k, String(v));
  }
  const qs = usp.toString();
  return qs ? `?${qs}` : "";
}

/** Polls an endpoint on an interval. Keeps the last good value on a failed
 * poll rather than clearing the UI, and reports connectivity separately. */
function usePolling<T>(path: string, intervalMs: number) {
  const [data, setData] = useState<T | null>(null);
  const [connected, setConnected] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    let cancelled = false;

    async function tick() {
      try {
        const result = await getJSON<T>(path);
        if (!cancelled) {
          setData(result);
          setConnected(true);
        }
      } catch {
        if (!cancelled) setConnected(false);
      } finally {
        if (!cancelled) {
          timer.current = window.setTimeout(tick, intervalMs);
        }
      }
    }

    tick();
    return () => {
      cancelled = true;
      window.clearTimeout(timer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, intervalMs]);

  return { data, connected };
}

// Polling intervals are deliberately modest with 4 cameras in view — four
// MJPEG streams already carry the "live" feel; JSON polling only needs to
// keep panels reasonably fresh, not frame-accurate.
export function useGlobalDetections(intervalMs = 700) {
  return usePolling<GlobalDetections>("/detections", intervalMs);
}

/** Live operator alert feed — GET /alerts (persistent, lifecycle-managed).
 * Renamed from useGlobalEvents: GET /events now means historical search,
 * not the live feed — see docs/PHASE3.md. */
export function useGlobalAlerts(intervalMs = 1200) {
  return usePolling<AlertItem[]>("/alerts", intervalMs);
}

export function useGlobalStatus(intervalMs = 2000) {
  return usePolling<GlobalStatus>("/status", intervalMs);
}

export function useCameraList(intervalMs = 2000) {
  return usePolling<CameraSummary[]>("/cameras", intervalMs);
}

export function cameraStreamUrl(cameraId: string): string {
  return `${API_BASE}/cameras/${encodeURIComponent(cameraId)}/stream`;
}

export async function restartCamera(cameraId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/cameras/${encodeURIComponent(cameraId)}/restart`, { method: "POST" });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? "restart failed");
}

export async function removeCamera(cameraId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/cameras/${encodeURIComponent(cameraId)}`, { method: "DELETE" });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? "remove failed");
}

export async function addVideoCamera(cameraName: string, file: File): Promise<CameraSummary> {
  const form = new FormData();
  form.append("camera_name", cameraName);
  form.append("source_type", "video");
  form.append("video", file);
  return submitAddCamera(form);
}

export async function addLiveCamera(cameraName: string, deviceIndex: number): Promise<CameraSummary> {
  const form = new FormData();
  form.append("camera_name", cameraName);
  form.append("source_type", "live");
  form.append("device_index", String(deviceIndex));
  return submitAddCamera(form);
}

async function submitAddCamera(form: FormData): Promise<CameraSummary> {
  const res = await fetch(`${API_BASE}/cameras`, { method: "POST", body: form });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.detail ?? "failed to add camera");
  }
  return body as CameraSummary;
}

export async function listCameraDevices(): Promise<CameraDevice[]> {
  return getJSON<CameraDevice[]>("/cameras/devices");
}

/** Pulls one camera's slice out of the already-polled global detections
 * payload — focus mode reuses this instead of opening a second poll loop. */
export function findCameraDetections(
  global: GlobalDetections | null,
  cameraId: string
): CameraDetectionState | null {
  if (!global) return null;
  return global.cameras.find((c) => c.camera_id === cameraId) ?? null;
}

// ---------------------------------------------------------------------------
// Phase 3 — Alert management
// ---------------------------------------------------------------------------
export async function acknowledgeAlert(alertId: string): Promise<AlertItem> {
  return sendJSON<AlertItem>(`/alerts/${encodeURIComponent(alertId)}/acknowledge`, "POST");
}

export async function resolveAlert(alertId: string): Promise<AlertItem> {
  return sendJSON<AlertItem>(`/alerts/${encodeURIComponent(alertId)}/resolve`, "POST");
}

// ---------------------------------------------------------------------------
// Phase 3 — Historical event search
// ---------------------------------------------------------------------------
export interface EventSearchFilters {
  camera_id?: string;
  event_type?: string;
  severity?: string;
  status?: string;
  identity?: string;
  start_time?: string;
  end_time?: string;
  limit?: number;
  offset?: number;
}

export async function searchEvents(filters: EventSearchFilters = {}): Promise<EventItem[]> {
  return getJSON<EventItem[]>(`/events${toQuery(filters)}`);
}

export async function fetchEvent(eventId: string): Promise<EventItem> {
  return getJSON<EventItem>(`/events/${encodeURIComponent(eventId)}`);
}

export function eventSnapshotUrl(eventId: string): string {
  return `${API_BASE}/events/${encodeURIComponent(eventId)}/snapshot`;
}

// ---------------------------------------------------------------------------
// Phase 3 — Investigation
// ---------------------------------------------------------------------------
export async function fetchEventInvestigation(eventId: string): Promise<EventInvestigation> {
  return getJSON<EventInvestigation>(`/investigations/event/${encodeURIComponent(eventId)}`);
}

export async function fetchPersonInvestigation(identity: string): Promise<PersonInvestigation> {
  return getJSON<PersonInvestigation>(`/investigations/person/${encodeURIComponent(identity)}`);
}

export async function fetchTrackInvestigation(cameraId: string, trackId: number): Promise<TrackInvestigation> {
  return getJSON<TrackInvestigation>(`/investigations/track/${encodeURIComponent(cameraId)}/${trackId}`);
}

// ---------------------------------------------------------------------------
// Phase 3 — Zone management
// ---------------------------------------------------------------------------
export async function fetchZones(cameraId?: string): Promise<ZoneItem[]> {
  return getJSON<ZoneItem[]>(`/zones${toQuery({ camera_id: cameraId })}`);
}

export async function createZone(
  cameraId: string,
  name: string,
  type: ZoneType,
  polygon: [number, number][]
): Promise<ZoneItem> {
  return sendJSON<ZoneItem>("/zones", "POST", { camera_id: cameraId, name, type, polygon, enabled: true });
}

export async function updateZone(
  zoneId: string,
  changes: Partial<{ name: string; type: ZoneType; polygon: [number, number][]; enabled: boolean }>
): Promise<ZoneItem> {
  return sendJSON<ZoneItem>(`/zones/${encodeURIComponent(zoneId)}`, "PUT", changes);
}

export async function deleteZone(zoneId: string): Promise<void> {
  await sendJSON(`/zones/${encodeURIComponent(zoneId)}`, "DELETE");
}
