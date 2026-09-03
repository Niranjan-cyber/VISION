import { useState } from "react";
import { addCamera } from "../api";
import type { CameraSummary } from "../types";

interface Props {
  onClose: () => void;
  onAdded: (camera: CameraSummary) => void;
  atLimit: boolean;
}

const ALLOWED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"];

export default function AddCameraModal({ onClose, onAdded, atLimit }: Props) {
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = name.trim().length > 0 && file !== null && !submitting && !atLimit;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit || !file) return;
    setSubmitting(true);
    setError(null);
    try {
      const camera = await addCamera(name.trim(), file);
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
          <span>ADD VIDEO SOURCE</span>
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

            <label className="modal-field">
              <span>Video File</span>
              <input
                type="file"
                accept={ALLOWED_EXTENSIONS.join(",")}
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              <span className="modal-hint">Allowed: {ALLOWED_EXTENSIONS.join(", ")}</span>
            </label>

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
