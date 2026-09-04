import { useEffect, useState } from "react";
import { addLiveCamera, addVideoCamera, listCameraDevices } from "../api";
import type { CameraDevice, CameraSummary, SourceType } from "../types";

interface Props {
  onClose: () => void;
  onAdded: (camera: CameraSummary) => void;
  atLimit: boolean;
}

const ALLOWED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"];

export default function AddCameraModal({ onClose, onAdded, atLimit }: Props) {
  const [sourceType, setSourceType] = useState<SourceType>("video");
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [devices, setDevices] = useState<CameraDevice[] | null>(null);
  const [devicesLoading, setDevicesLoading] = useState(false);
  const [deviceIndex, setDeviceIndex] = useState<number | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (sourceType !== "live" || devices !== null || atLimit) return;
    setDevicesLoading(true);
    listCameraDevices()
      .then((found) => {
        setDevices(found);
        const firstAvailable = found.find((d) => d.available);
        if (firstAvailable) setDeviceIndex(firstAvailable.device_index);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to list camera devices"))
      .finally(() => setDevicesLoading(false));
  }, [sourceType, devices, atLimit]);

  const availableDevices = devices?.filter((d) => d.available) ?? [];
  const canSubmit =
    name.trim().length > 0 &&
    !submitting &&
    !atLimit &&
    (sourceType === "video" ? file !== null : deviceIndex !== null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const camera =
        sourceType === "video"
          ? await addVideoCamera(name.trim(), file as File)
          : await addLiveCamera(name.trim(), deviceIndex as number);
      onAdded(camera);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add camera");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <span>ADD CAMERA SOURCE</span>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        {atLimit ? (
          <div className="modal-body">
            <div className="limit-banner">Maximum 4 active camera streams reached. Remove a camera before adding another.</div>
          </div>
        ) : (
          <form className="modal-body" onSubmit={handleSubmit}>
            <div className="source-type-toggle" role="radiogroup" aria-label="Camera source">
              <label className={`source-type-option ${sourceType === "video" ? "active" : ""}`}>
                <input
                  type="radio"
                  name="source_type"
                  value="video"
                  checked={sourceType === "video"}
                  onChange={() => setSourceType("video")}
                />
                Upload Video
              </label>
              <label className={`source-type-option ${sourceType === "live" ? "active" : ""}`}>
                <input
                  type="radio"
                  name="source_type"
                  value="live"
                  checked={sourceType === "live"}
                  onChange={() => setSourceType("live")}
                />
                Live Camera
              </label>
            </div>

            <label className="modal-field">
              <span>Camera Name</span>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. North Perimeter"
                autoFocus
              />
            </label>

            {sourceType === "video" ? (
              <label className="modal-field">
                <span>Video File</span>
                <input
                  type="file"
                  accept={ALLOWED_EXTENSIONS.join(",")}
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
                <span className="modal-hint">Allowed: {ALLOWED_EXTENSIONS.join(", ")}</span>
              </label>
            ) : (
              <label className="modal-field">
                <span>Source</span>
                {devicesLoading && <span className="modal-hint">Scanning local camera devices…</span>}
                {!devicesLoading && availableDevices.length === 0 && devices !== null && (
                  <span className="modal-hint modal-hint-warn">
                    No local camera devices found. Make sure a webcam/USB camera is connected and not in use
                    by another application, then reopen this dialog.
                  </span>
                )}
                {availableDevices.length > 0 && (
                  <select
                    value={deviceIndex ?? ""}
                    onChange={(e) => setDeviceIndex(Number(e.target.value))}
                  >
                    {availableDevices.map((d) => (
                      <option key={d.device_index} value={d.device_index}>
                        Camera {d.device_index} ({d.width}x{d.height})
                      </option>
                    ))}
                  </select>
                )}
              </label>
            )}

            {error && <div className="modal-error">{error}</div>}

            <div className="modal-actions">
              <button type="button" className="modal-cancel" onClick={onClose}>
                Cancel
              </button>
              <button type="submit" className="modal-submit" disabled={!canSubmit}>
                {submitting ? "Adding…" : "Add Camera"}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
