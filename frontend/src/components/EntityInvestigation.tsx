import { useEffect, useState } from "react";
import { ArrowLeft, Car, UserRound } from "lucide-react";
import { fetchPersonInvestigation, fetchTrackInvestigation } from "../api";
import type { PersonInvestigation, TrackInvestigation } from "../types";
import EventTimeline from "./EventTimeline";

export type EntityQuery = { type: "person"; identity: string } | { type: "track"; cameraId: string; trackId: number };

interface Props {
  query: EntityQuery;
  onBack: () => void;
  onSelectEvent: (eventId: string) => void;
}

function formatWallClock(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString("en-GB", { hour12: false });
}

/** Slice 3.5 — person/vehicle investigation. A "person" query aggregates a
 * *recognized* identity across every camera it appeared on; a "track" query
 * (unrecognized person, or any vehicle — the pipeline has no name for
 * either) is scoped to the one (camera, track) it was actually observed on.
 * Never claims a cross-camera identity the system didn't establish. */
export default function EntityInvestigation({ query, onBack, onSelectEvent }: Props) {
  const [person, setPerson] = useState<PersonInvestigation | null>(null);
  const [track, setTrack] = useState<TrackInvestigation | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setPerson(null);
    setTrack(null);
    setError(null);
    if (query.type === "person") {
      fetchPersonInvestigation(query.identity)
        .then(setPerson)
        .catch((err) => setError(err instanceof Error ? err.message : "Failed to load"));
    } else {
      fetchTrackInvestigation(query.cameraId, query.trackId)
        .then(setTrack)
        .catch((err) => setError(err instanceof Error ? err.message : "Failed to load"));
    }
  }, [query]);

  const isVehicle = track?.object_type && track.object_type !== "person";

  return (
    <div className="investigation">
      <div className="camera-detail-head">
        <button className="back-btn" onClick={onBack}>
          <ArrowLeft size={14} strokeWidth={2.25} />
          Back
        </button>
        <div className="camera-detail-title">Entity Investigation</div>
      </div>

      {error && (
        <div className="panel">
          <div className="empty-state">{error}</div>
        </div>
      )}

      {person && (
        <div className="panel entity-panel">
          <div className="panel-header">
            <span>
              <UserRound size={14} strokeWidth={2.25} className="entity-header-icon" />
              Person
            </span>
          </div>
          <div className="entity-headline">{person.identity}</div>
          <div className="entity-facts">
            <div className="entity-fact">
              <span className="entity-fact-label">Identity</span>
              <span className="status-badge status-ok">RECOGNIZED</span>
            </div>
            <div className="entity-fact">
              <span className="entity-fact-label">Last Seen</span>
              <span className="mono">{formatWallClock(person.last_seen)}</span>
            </div>
            <div className="entity-fact">
              <span className="entity-fact-label">Cameras</span>
              <span className="mono">{person.cameras.join(", ")}</span>
            </div>
          </div>
        </div>
      )}

      {track && (
        <div className="panel entity-panel">
          <div className="panel-header">
            <span>
              {isVehicle ? (
                <Car size={14} strokeWidth={2.25} className="entity-header-icon" />
              ) : (
                <UserRound size={14} strokeWidth={2.25} className="entity-header-icon" />
              )}
              {isVehicle ? "Vehicle" : "Person"}
            </span>
          </div>
          <div className="entity-headline mono">Track #{track.track_id}</div>
          <div className="entity-facts">
            <div className="entity-fact">
              <span className="entity-fact-label">Identity</span>
              {isVehicle ? (
                <span>N/A — VEHICLE</span>
              ) : (
                <span className="status-badge status-warn">NOT RECOGNIZED</span>
              )}
            </div>
            <div className="entity-fact">
              <span className="entity-fact-label">Camera</span>
              <span className="mono">{track.camera_id}</span>
            </div>
            <div className="entity-fact">
              <span className="entity-fact-label">Last Seen</span>
              <span className="mono">{formatWallClock(track.last_seen)}</span>
            </div>
            {isVehicle && (
              <div className="entity-fact">
                <span className="entity-fact-label">Plate</span>
                <span>{track.plate ?? "NOT AVAILABLE"}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {(person || track) && (
        <EventTimeline
          events={person?.events ?? track?.events ?? []}
          onSelectEvent={onSelectEvent}
          title="Events"
        />
      )}
    </div>
  );
}
