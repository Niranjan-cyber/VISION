import { useEffect, useRef, useState } from "react";
import { Pencil, Plus, Power, Trash2 } from "lucide-react";
import { cameraStreamUrl, createZone, deleteZone, fetchZones, updateZone } from "../api";
import type { CameraSummary, ZoneItem, ZoneType } from "../types";

interface Props {
  cameras: CameraSummary[];
}

type DraftRect = { x1: number; y1: number; x2: number; y2: number } | null;

const ZONE_TYPES: ZoneType[] = ["restricted", "warning", "monitored"];

export default function ZoneManagement({ cameras }: Props) {
  const [cameraId, setCameraId] = useState(cameras[0]?.camera_id ?? "");
  const [zones, setZones] = useState<ZoneItem[]>([]);
  const [loading, setLoading] = useState(false);

  const [drawing, setDrawing] = useState(false);
  const [draft, setDraft] = useState<DraftRect>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [pendingName, setPendingName] = useState("");
  const [pendingType, setPendingType] = useState<ZoneType>("restricted");
  const [editingZoneId, setEditingZoneId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (!cameraId && cameras[0]) setCameraId(cameras[0].camera_id);
  }, [cameras, cameraId]);

  async function loadZones() {
    if (!cameraId) return;
    setLoading(true);
    try {
      setZones(await fetchZones(cameraId));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadZones();
    setDrawing(false);
    setDraft(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraId]);

  function relativePoint(e: React.MouseEvent): { x: number; y: number } | null {
    const img = imgRef.current;
    if (!img || !img.naturalWidth) return null;
    const rect = img.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    const x = Math.max(0, Math.min(img.naturalWidth, (e.clientX - rect.left) * scaleX));
    const y = Math.max(0, Math.min(img.naturalHeight, (e.clientY - rect.top) * scaleY));
    return { x: Math.round(x), y: Math.round(y) };
  }

  function handleMouseDown(e: React.MouseEvent) {
    if (!drawing) return;
    const pt = relativePoint(e);
    if (!pt) return;
    setDragStart(pt);
    setDraft({ x1: pt.x, y1: pt.y, x2: pt.x, y2: pt.y });
  }

  function handleMouseMove(e: React.MouseEvent) {
    if (!drawing || !dragStart) return;
    const pt = relativePoint(e);
    if (!pt) return;
    setDraft({ x1: dragStart.x, y1: dragStart.y, x2: pt.x, y2: pt.y });
  }

  function handleMouseUp() {
    if (!drawing || !draft) return;
    setDragStart(null);
    // Require a minimally-sized rectangle so a stray click doesn't open the form.
    if (Math.abs(draft.x2 - draft.x1) < 8 || Math.abs(draft.y2 - draft.y1) < 8) {
      setDraft(null);
    }
  }

  function rectToPolygon(r: NonNullable<DraftRect>): [number, number][] {
    const x1 = Math.min(r.x1, r.x2);
    const x2 = Math.max(r.x1, r.x2);
    const y1 = Math.min(r.y1, r.y2);
    const y2 = Math.max(r.y1, r.y2);
    return [
      [x1, y1],
      [x2, y1],
      [x2, y2],
      [x1, y2],
    ];
  }

  async function handleSaveDraft() {
    if (!draft || !pendingName.trim()) return;
    setError(null);
    try {
      if (editingZoneId) {
        await updateZone(editingZoneId, { name: pendingName.trim(), type: pendingType, polygon: rectToPolygon(draft) });
      } else {
        await createZone(cameraId, pendingName.trim(), pendingType, rectToPolygon(draft));
      }
      cancelDrawing();
      await loadZones();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save zone");
    }
  }

  function startCreate() {
    setEditingZoneId(null);
    setPendingName("");
    setPendingType("restricted");
    setDraft(null);
    setDrawing(true);
  }

  function startEdit(zone: ZoneItem) {
    setEditingZoneId(zone.id);
    setPendingName(zone.name);
    setPendingType(zone.type);
    const xs = zone.polygon.map((p) => p[0]);
    const ys = zone.polygon.map((p) => p[1]);
    setDraft({ x1: Math.min(...xs), y1: Math.min(...ys), x2: Math.max(...xs), y2: Math.max(...ys) });
    setDrawing(true);
  }

  function cancelDrawing() {
    setDrawing(false);
    setDraft(null);
    setDragStart(null);
    setEditingZoneId(null);
  }

  async function handleDelete(zoneId: string) {
    await deleteZone(zoneId);
    await loadZones();
  }

  async function handleToggleEnabled(zone: ZoneItem) {
    await updateZone(zone.id, { enabled: !zone.enabled });
    await loadZones();
  }

  const camera = cameras.find((c) => c.camera_id === cameraId);

  return (
    <>
      <div className="panel">
        <div className="panel-header">
          <span>Zone Management</span>
          <select className="zone-camera-select" value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
            {cameras.map((c) => (
              <option key={c.camera_id} value={c.camera_id}>
                {c.camera_id} — {c.camera_name}
              </option>
            ))}
          </select>
        </div>

        {camera && camera.status === "online" ? (
          <div className="zone-editor-frame">
            <img
              ref={imgRef}
              className="zone-editor-image"
              src={`${cameraStreamUrl(cameraId)}?t=${Date.now()}`}
              alt={`${camera.camera_name} feed`}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              draggable={false}
            />
            <svg className="zone-editor-overlay" preserveAspectRatio="none">
              {zones.map((z) => (
                <ZonePolygon key={z.id} zone={z} imgRef={imgRef} />
              ))}
              {draft && <DraftRectSvg draft={draft} imgRef={imgRef} />}
            </svg>
            {drawing && !draft && <div className="zone-editor-hint">Click and drag on the feed to draw a zone</div>}
          </div>
        ) : (
          <div className="empty-state">Camera feed unavailable — select an online camera.</div>
        )}

        {draft && (
          <div className="zone-draft-form">
            <input
              type="text"
              placeholder="Zone name"
              value={pendingName}
              onChange={(e) => setPendingName(e.target.value)}
              autoFocus
            />
            <select value={pendingType} onChange={(e) => setPendingType(e.target.value as ZoneType)}>
              {ZONE_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t.toUpperCase()}
                </option>
              ))}
            </select>
            <button className="modal-cancel" onClick={cancelDrawing}>
              Cancel
            </button>
            <button className="modal-submit" disabled={!pendingName.trim()} onClick={handleSaveDraft}>
              {editingZoneId ? "Save Changes" : "Create Zone"}
            </button>
          </div>
        )}
        {error && <div className="modal-error zone-error">{error}</div>}
      </div>

      <div className="panel">
        <div className="panel-header">
          <span>Zones — {cameraId}</span>
          {!drawing && (
            <button className="add-camera-btn" onClick={startCreate} disabled={!camera}>
              <Plus size={14} strokeWidth={2.5} />
              Add Zone
            </button>
          )}
        </div>
        {loading ? (
          <div className="empty-state">Loading zones…</div>
        ) : zones.length === 0 ? (
          <div className="empty-state">NO ZONE CONFIGURED</div>
        ) : (
          <div className="management-list zone-list">
            {zones.map((z) => (
              <div className="management-row zone-row" key={z.id}>
                <span className={`dot ${z.enabled ? "dot-ok" : "dot-off"}`} />
                <span className="zone-row-name">{z.name}</span>
                <span className="zone-type-badge">{z.type.toUpperCase()}</span>
                <span className="mgmt-status">{z.enabled ? "ACTIVE" : "DISABLED"}</span>
                <span className="mgmt-actions">
                  <button onClick={() => startEdit(z)}>
                    <Pencil size={12} strokeWidth={2.25} />
                    Edit
                  </button>
                  <button onClick={() => handleToggleEnabled(z)}>
                    <Power size={12} strokeWidth={2.25} />
                    {z.enabled ? "Disable" : "Enable"}
                  </button>
                  <button className="mgmt-remove" onClick={() => handleDelete(z.id)}>
                    <Trash2 size={12} strokeWidth={2.25} />
                    Delete
                  </button>
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

function toScreenRect(
  r: { x1: number; y1: number; x2: number; y2: number },
  imgRef: React.RefObject<HTMLImageElement | null>
) {
  const img = imgRef.current;
  if (!img || !img.naturalWidth) return null;
  const scaleX = 100 / img.naturalWidth;
  const scaleY = 100 / img.naturalHeight;
  return {
    x: Math.min(r.x1, r.x2) * scaleX,
    y: Math.min(r.y1, r.y2) * scaleY,
    w: Math.abs(r.x2 - r.x1) * scaleX,
    h: Math.abs(r.y2 - r.y1) * scaleY,
  };
}

function ZonePolygon({ zone, imgRef }: { zone: ZoneItem; imgRef: React.RefObject<HTMLImageElement | null> }) {
  const img = imgRef.current;
  if (!img || !img.naturalWidth) return null;
  const scaleX = 100 / img.naturalWidth;
  const scaleY = 100 / img.naturalHeight;
  const points = zone.polygon.map(([x, y]) => `${x * scaleX}%,${y * scaleY}%`).join(" ");
  const cx = zone.polygon.reduce((s, p) => s + p[0], 0) / zone.polygon.length;
  const cy = Math.min(...zone.polygon.map((p) => p[1]));
  return (
    <g className={`zone-polygon ${zone.enabled ? "" : "zone-polygon-disabled"}`}>
      <polygon points={points} />
      <text x={`${cx * scaleX}%`} y={`${cy * scaleY}%`} dy="-6">
        {zone.name.toUpperCase()}
      </text>
    </g>
  );
}

function DraftRectSvg({ draft, imgRef }: { draft: NonNullable<DraftRect>; imgRef: React.RefObject<HTMLImageElement | null> }) {
  const r = toScreenRect(draft, imgRef);
  if (!r) return null;
  return <rect className="zone-draft-rect" x={`${r.x}%`} y={`${r.y}%`} width={`${r.w}%`} height={`${r.h}%`} />;
}
