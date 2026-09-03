import Header from "./components/Header";
import SystemStatusStrip from "./components/SystemStatusStrip";
import VideoPanel from "./components/VideoPanel";
import AlertsPanel from "./components/AlertsPanel";
import StatsBar from "./components/StatsBar";
import PersonsTable from "./components/PersonsTable";
import VehiclesTable from "./components/VehiclesTable";
import { useDetections, useEvents, useSystemStatus } from "./api";

export default function App() {
  const { data: detections, connected: detectionsConnected } = useDetections();
  const { data: events } = useEvents();
  const { data: status, connected: statusConnected } = useSystemStatus();

  return (
    <div className="app-shell">
      <Header status={status} connected={statusConnected} />
      <SystemStatusStrip status={status} connected={statusConnected} />

      <div className="main-grid">
        <VideoPanel />
        <AlertsPanel alerts={events ?? []} />
      </div>

      <StatsBar stats={detections?.statistics ?? null} />

      <div className="main-grid">
        <PersonsTable persons={detections?.persons ?? []} />
        <VehiclesTable
          vehicles={detections?.vehicles ?? []}
          anprEnabled={detections?.anpr_enabled ?? false}
        />
      </div>

      {!detectionsConnected && (
        <div className="conn-banner">
          Waiting for backend connection — is the FastAPI server running?
        </div>
      )}
    </div>
  );
}
