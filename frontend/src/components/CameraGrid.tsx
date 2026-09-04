import CameraCard from "./CameraCard";
import type { AlertItem, CameraDetectionState } from "../types";

interface Props {
  cameras: CameraDetectionState[];
  alerts: AlertItem[];
  onFocusCamera: (cameraId: string) => void;
  maxSlots: number;
}

export default function CameraGrid({ cameras, alerts, onFocusCamera, maxSlots }: Props) {
  const emptySlots = Math.max(0, maxSlots - cameras.length);

  return (
    <div className="panel camera-grid-panel">
      <div className="panel-header">
        <span>SURVEILLANCE GRID — {cameras.length}/{maxSlots} CAMERAS</span>
      </div>
      <div className="camera-grid">
        {cameras.map((cam) => (
          <CameraCard
            key={cam.camera_id}
            camera={cam}
            alerts={alerts.filter((a) => a.camera_id === cam.camera_id)}
            onFocus={() => onFocusCamera(cam.camera_id)}
          />
        ))}
        {Array.from({ length: emptySlots }).map((_, i) => (
          <div className="camera-card-empty" key={`empty-${i}`}>
            <span>No camera in this slot</span>
            <span className="empty-hint">Use Add Camera to fill it</span>
          </div>
        ))}
      </div>
    </div>
  );
}
