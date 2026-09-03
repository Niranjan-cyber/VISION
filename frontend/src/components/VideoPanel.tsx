import { useState } from "react";
import { streamUrl } from "../api";

export default function VideoPanel() {
  const [loaded, setLoaded] = useState(false);
  // Cache-bust only once per mount — the MJPEG stream itself is continuous,
  // this just avoids the browser serving a stale cached single frame.
  const [src] = useState(() => `${streamUrl()}?t=${Date.now()}`);

  return (
    <div className="panel video-panel">
      <div className="panel-header">
        <span>LIVE SURVEILLANCE</span>
      </div>
      <div className="video-frame">
        {!loaded && <div className="video-placeholder">Connecting to camera feed…</div>}
        <img
          src={src}
          alt="Live annotated surveillance feed"
          onLoad={() => setLoaded(true)}
          onError={() => setLoaded(false)}
          style={{ display: loaded ? "block" : "none" }}
        />
      </div>
    </div>
  );
}
