import { useEffect, useState } from "react";
import { Search } from "lucide-react";
import { searchEvents } from "../api";
import type { CameraSummary, EventItem } from "../types";
import EventTimeline from "./EventTimeline";

interface Props {
  cameras: CameraSummary[];
  onSelectEvent: (eventId: string) => void;
}

const EVENT_TYPES = ["INTRUSION", "UNKNOWN_PERSON_INTRUSION", "LOITERING", "SUSPICIOUS_VEHICLE"];
const SEVERITIES = ["HIGH", "MEDIUM", "LOW"];
const STATUSES = ["NEW", "ACKNOWLEDGED", "RESOLVED"];

export default function EventHistory({ cameras, onSelectEvent }: Props) {
  const [cameraId, setCameraId] = useState("");
  const [eventType, setEventType] = useState("");
  const [severity, setSeverity] = useState("");
  const [status, setStatus] = useState("");
  const [events, setEvents] = useState<EventItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runSearch() {
    setLoading(true);
    setError(null);
    try {
      const results = await searchEvents({
        camera_id: cameraId || undefined,
        event_type: eventType || undefined,
        severity: severity || undefined,
        status: status || undefined,
        limit: 100,
      });
      setEvents(results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  // Load an unfiltered page on first mount so the page never opens empty.
  useEffect(() => {
    runSearch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <div className="panel event-search-panel">
        <div className="panel-header">
          <span>Search Filters</span>
        </div>
        <div className="event-search-body">
          <label className="event-search-field">
            <span>Camera</span>
            <select value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
              <option value="">All Cameras</option>
              {cameras.map((c) => (
                <option key={c.camera_id} value={c.camera_id}>
                  {c.camera_id} — {c.camera_name}
                </option>
              ))}
            </select>
          </label>
          <label className="event-search-field">
            <span>Type</span>
            <select value={eventType} onChange={(e) => setEventType(e.target.value)}>
              <option value="">All Events</option>
              {EVENT_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          </label>
          <label className="event-search-field">
            <span>Severity</span>
            <select value={severity} onChange={(e) => setSeverity(e.target.value)}>
              <option value="">All</option>
              {SEVERITIES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label className="event-search-field">
            <span>Status</span>
            <select value={status} onChange={(e) => setStatus(e.target.value)}>
              <option value="">All</option>
              {STATUSES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <button className="event-search-btn" onClick={runSearch} disabled={loading}>
            <Search size={14} strokeWidth={2.25} />
            {loading ? "Searching…" : "Search"}
          </button>
        </div>
        {error && <div className="modal-error event-search-error">{error}</div>}
      </div>

      <EventTimeline
        events={events ?? []}
        onSelectEvent={onSelectEvent}
        title={events ? `${events.length} Event${events.length === 1 ? "" : "s"} Found` : "Event History"}
      />
    </>
  );
}
