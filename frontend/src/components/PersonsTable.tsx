import type { Person } from "../types";

interface Props {
  persons: Person[];
}

/**
 * Identity has three genuinely distinct states — collapsing them would be
 * misleading (a face that was never seen is not the same as a face that was
 * seen and didn't match). Never use words like FLAGGED/THREAT/CONFIRMED here;
 * those belong to real EventEngine-generated alerts, not a bare face match.
 */
function identityDisplay(p: Person): { text: string; statusText: string; statusClass: string } {
  if (p.identity === null) {
    return { text: "NO FACE DETECTED", statusText: "UNVERIFIED", statusClass: "status-neutral" };
  }
  if (p.identity === "UNKNOWN") {
    return { text: "NO MATCH", statusText: "NOT RECOGNIZED", statusClass: "status-warn" };
  }
  return { text: p.identity, statusText: "RECOGNIZED", statusClass: "status-ok" };
}

export default function PersonsTable({ persons }: Props) {
  return (
    <div className="panel table-panel">
      <div className="panel-header">
        <span>Tracked Persons</span>
      </div>
      {persons.length === 0 ? (
        <div className="empty-state">No persons currently tracked.</div>
      ) : (
        <div className="person-cards">
          {persons.map((p) => {
            const id = identityDisplay(p);
            return (
              <div className="person-card" key={p.track_id}>
                <div className="person-card-row">
                  <span className="mono track-id">#{String(p.track_id).padStart(2, "0")}</span>
                  <span className={`status-badge ${id.statusClass}`}>{id.statusText}</span>
                </div>
                <div className="person-identity">{id.text}</div>
                <div className="person-card-row detail-row">
                  {p.face_similarity !== null && (
                    <span>
                      Match: <span className="mono">{Math.round(p.face_similarity * 100)}%</span>
                    </span>
                  )}
                  <span>
                    Zone: <span className="mono">{p.zone ?? "—"}</span>
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
