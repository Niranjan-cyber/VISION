import type { Vehicle } from "../types";

interface Props {
  vehicles: Vehicle[];
  anprEnabled: boolean;
}

export default function VehiclesTable({ vehicles, anprEnabled }: Props) {
  return (
    <div className="panel table-panel">
      <div className="panel-header">
        <span>VEHICLES {anprEnabled ? "/ ANPR" : ""}</span>
        {!anprEnabled && <span className="badge-muted">ANPR OFFLINE</span>}
      </div>
      {!anprEnabled ? (
        <div className="empty-state">
          Automatic plate recognition is disabled for this demo — no verified-legible
          plate was available in the source footage. Vehicle detection/tracking is
          still shown below when present.
        </div>
      ) : null}
      {vehicles.length === 0 ? (
        <div className="empty-state">No vehicles currently tracked.</div>
      ) : (
        <div className="person-cards">
          {vehicles.map((v) => (
            <div className="person-card" key={v.track_id}>
              <div className="person-card-row">
                <span className="mono track-id">#{String(v.track_id).padStart(2, "0")}</span>
                <span className="status-badge status-neutral">{v.type.toUpperCase()}</span>
              </div>
              {anprEnabled && (
                <div className="person-identity">
                  {v.plate ?? "PLATE NOT READ"}
                </div>
              )}
              <div className="person-card-row detail-row">
                {anprEnabled && v.plate_confidence !== null && v.plate_confidence !== undefined && (
                  <span>
                    OCR: <span className="mono">{Math.round(v.plate_confidence * 100)}%</span>
                  </span>
                )}
                <span>
                  Zone: <span className="mono">{v.zone ?? "—"}</span>
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
