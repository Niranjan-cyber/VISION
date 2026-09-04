import { useEffect, useRef, useState } from "react";
import type {
  AlertItem,
  CameraDetectionState,
  CameraDevice,
  CameraSummary,
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

export function useGlobalEvents(intervalMs = 1200) {
  return usePolling<AlertItem[]>("/events", intervalMs);
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
