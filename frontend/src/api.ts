import { useEffect, useRef, useState } from "react";
import type { AlertItem, DetectionState, SystemStatus } from "./types";

// Same-origin by default (works when the backend serves the build), override
// for local dev against a separately-running backend.
export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `${res.status} ${res.statusText}`);
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

export function useDetections(intervalMs = 500) {
  return usePolling<DetectionState>("/detections", intervalMs);
}

export function useEvents(intervalMs = 1000) {
  return usePolling<AlertItem[]>("/events", intervalMs);
}

export function useSystemStatus(intervalMs = 2000) {
  return usePolling<SystemStatus>("/status", intervalMs);
}

export function streamUrl(): string {
  return `${API_BASE}/stream`;
}

export async function restartDemo(): Promise<void> {
  await fetch(`${API_BASE}/restart`, { method: "POST" });
}
