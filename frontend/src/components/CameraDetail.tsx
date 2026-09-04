import VideoFeed from "./VideoFeed";
import PersonsTable from "./PersonsTable";
import VehiclesTable from "./VehiclesTable";
import { cameraStreamUrl } from "../api";
import type { AlertItem, CameraDetectionState } from "../types";

interface Props {
  camera: CameraDetectionState;
  alerts: AlertItem[]; // this camera's alerts only
  onBack: () => void;
}

export default function CameraDetail({ camera, alerts, onBack }: Props) {
  return (
    <div className="camera-detail">
      <div className="camera-detail-head">
        <button className="back-btn" onClick={onBack}>
          ← All Cameras
        </button>
        <div className="camera-detail-title">
          <span className="mono">{camera.camera_id}</span> — {camera.camera_name.toUpperCase()}
          <span className={`source-type-badge source-type-${camera.source_type}`}>
            {camera.source_type === "live" ? "● LIVE" : "RECORDED"}
          </span>
          {!camera.has_zone && <span className="no-zone-badge">NO ZONE CONFIGURED</span>}
        </div>
        <span className={`cam-status-pill cam-status-${camera.camera_status}`}>
          <span className="dot" />
          {camera.camera_status.toUpperCase()}
        </span>
      </div>

      <div className="main-grid">
        <div className="panel video-panel">
          <div className="panel-header">
            <span>SURVEILLANCE FEED</span>
          </div>
          {camera.camera_status === "online" ? (
            <VideoFeed streamUrl={cameraStreamUrl(camera.camera_id)} alt={`Live feed — ${camera.camera_name}`} />
          ) : (
            <div className="video-frame">
              <div className="video-placeholder">
                {camera.camera_status === "error" ? "Processing error" : camera.camera_status}
              </div>
            </div>
          )}
        </div>

        <div className="panel alerts-panel">
          <div className="panel-header">
            <span>ALERTS — {camera.camera_name.toUpperCase()}</span>
          </div>
          <div className="alerts-list">
            {alerts.length === 0 && <div className="empty-state">No active alerts for this camera.</div>}
            {alerts.map((a) => (
              <div className={`alert-card sev-${a.severity.toLowerCase()}`} key={a.alert_id}>
                <div className="alert-top">
                  <span className="alert-severity">{a.severity} PRIORITY</span>
                </div>
                <div className="alert-title">{a.title.replace(/[^\x00-\x7F]/g, "").trim()}</div>
                {a.zone_name && <div className="alert-zone">{a.zone_name}</div>}
                {a.track_id !== null && <div className="alert-meta mono">Track #{String(a.track_id).padStart(2, "0")}</div>}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="stats-bar">
        <div className="stat-card"><div className="stat-value">{camera.statistics.persons}</div><div className="stat-label">PERSONS</div></div>
        <div className="stat-card"><div className="stat-value">{camera.statistics.vehicles}</div><div className="stat-label">VEHICLES</div></div>
        <div className="stat-card"><div className="stat-value">{camera.statistics.faces_detected}</div><div className="stat-label">FACES</div></div>
        <div className="stat-card"><div className="stat-value">{camera.statistics.recognized_faces}</div><div className="stat-label">IDENTIFIED</div></div>
        <div className="stat-card"><div className="stat-value">{camera.statistics.active_events}</div><div className="stat-label">ACTIVE EVENTS</div></div>
      </div>

      <div className="main-grid">
        <PersonsTable persons={camera.persons} />
        <VehiclesTable vehicles={camera.vehicles} anprEnabled={camera.anpr_enabled} />
      </div>
    </div>
  );
}
