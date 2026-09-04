import { useState } from "react";
import { Plus, RotateCw, Trash2 } from "lucide-react";
import type { CameraSummary } from "../types";
import { removeCamera, restartCamera } from "../api";

interface Props {
  cameras: CameraSummary[];
  maxSlots: number;
  onFocusCamera: (cameraId: string) => void;
  onAddClick: () => void;
}

export default function CameraManagement({ cameras, maxSlots, onFocusCamera, onAddClick }: Props) {
  const [busy, setBusy] = useState<string | null>(null);

  async function handleRestart(id: string) {
    setBusy(id);
    try {
      await restartCamera(id);
    } finally {
      setBusy(null);
    }
  }

  async function handleRemove(id: string) {
    setBusy(id);
    try {
      await removeCamera(id);
    } finally {
      setBusy(null);
    }
  }

  const atLimit = cameras.length >= maxSlots;

  return (
    <div className="panel management-panel">
      <div className="panel-header">
        <span>Camera Management</span>
        <button className="add-camera-btn" onClick={onAddClick} disabled={atLimit} title={atLimit ? `Maximum ${maxSlots} active camera streams reached.` : undefined}>
          <Plus size={14} strokeWidth={2.5} />
          Add Camera
        </button>
      </div>
      {atLimit && (
        <div className="limit-banner">Maximum {maxSlots} active camera streams reached. Remove a camera to add another.</div>
      )}
      <div className="management-list">
        {cameras.map((cam) => (
          <div className="management-row" key={cam.camera_id}>
            <span className={`dot dot-${cam.status === "online" ? "ok" : cam.status === "error" ? "bad" : "off"}`} />
            <span className="mono mgmt-id">{cam.camera_id}</span>
            <button className="mgmt-name-btn" onClick={() => onFocusCamera(cam.camera_id)}>
              {cam.camera_name}
            </button>
            <span className={`source-type-badge source-type-${cam.source_type}`}>
              {cam.source_type === "live" ? "● LIVE" : "RECORDED"}
            </span>
            <span className="mgmt-source mono">{cam.video_source.split(/[\\/]/).pop()}</span>
            <span className="mgmt-status">{cam.status.toUpperCase()}</span>
            <span className="mgmt-actions">
              <button disabled={busy === cam.camera_id} onClick={() => handleRestart(cam.camera_id)}>
                <RotateCw size={12} strokeWidth={2.25} />
                Restart
              </button>
              <button disabled={busy === cam.camera_id} className="mgmt-remove" onClick={() => handleRemove(cam.camera_id)}>
                <Trash2 size={12} strokeWidth={2.25} />
                Remove
              </button>
            </span>
          </div>
        ))}
        {cameras.length === 0 && <div className="empty-state">No cameras configured.</div>}
      </div>
    </div>
  );
}
