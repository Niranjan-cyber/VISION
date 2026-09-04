import { AlertTriangle, ScanFace, ShieldCheck, Users, Car } from "lucide-react";
import type { CameraDetectionState, GlobalDetections } from "../types";

interface Props {
  detections: GlobalDetections | null;
}

function alertCountFor(cam: CameraDetectionState): number {
  return cam.statistics.active_events;
}

export default function GlobalStats({ detections }: Props) {
  const stats = detections?.statistics ?? null;
  const items: { label: string; value: number | string; icon: typeof Users }[] = [
    { label: "PERSONS DETECTED", value: stats?.persons ?? "—", icon: Users },
    { label: "VEHICLES DETECTED", value: stats?.vehicles ?? "—", icon: Car },
    { label: "FACES DETECTED", value: stats?.faces_detected ?? "—", icon: ScanFace },
    { label: "IDENTIFIED", value: stats?.recognized_faces ?? "—", icon: ShieldCheck },
    { label: "ACTIVE ALERTS", value: stats?.active_events ?? "—", icon: AlertTriangle },
  ];

  return (
    <div className="panel stats-panel">
      <div className="panel-header">
        <span>Global Statistics</span>
        {stats && <span className="tag">{stats.cameras_active} camera{stats.cameras_active === 1 ? "" : "s"} active</span>}
      </div>
      <div className="stats-bar">
        {items.map((it) => (
          <div className="stat-card" key={it.label}>
            <it.icon className="stat-icon" size={16} strokeWidth={2} />
            <div className="stat-value">{it.value}</div>
            <div className="stat-label">{it.label}</div>
          </div>
        ))}
      </div>
      {detections && detections.cameras.length > 0 && (
        <div className="cam-breakdown">
          {detections.cameras.map((cam) => (
            <div className="cam-breakdown-row" key={cam.camera_id}>
              <span className="mono cam-breakdown-id">{cam.camera_id}</span>
              <span className="cam-breakdown-name">{cam.camera_name}</span>
              <span className="cam-breakdown-metric">{cam.statistics.persons} person{cam.statistics.persons === 1 ? "" : "s"}</span>
              <span className={`cam-breakdown-metric ${alertCountFor(cam) > 0 ? "cam-breakdown-alert" : ""}`}>
                {alertCountFor(cam)} alert{alertCountFor(cam) === 1 ? "" : "s"}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
