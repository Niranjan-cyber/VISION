import { useState } from "react";

interface Props {
  streamUrl: string;
  alt: string;
  compact?: boolean;
}

/** Bare MJPEG video element — no panel chrome. Used inside both CameraCard
 * (small tile) and CameraDetail (large focus view) so there is exactly one
 * implementation of "render a camera's live stream". */
export default function VideoFeed({ streamUrl, alt, compact }: Props) {
  const [loaded, setLoaded] = useState(false);
  // Cache-bust only once per mount — the MJPEG stream itself is continuous,
  // this just avoids the browser serving a stale cached single frame.
  const [src] = useState(() => `${streamUrl}?t=${Date.now()}`);

  return (
    <div className={`video-frame ${compact ? "video-frame-compact" : ""}`}>
      {!loaded && <div className="video-placeholder">Connecting…</div>}
      <img
        src={src}
        alt={alt}
        onLoad={() => setLoaded(true)}
        onError={() => setLoaded(false)}
        style={{ display: loaded ? "block" : "none" }}
      />
    </div>
  );
}
