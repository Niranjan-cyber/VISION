import math
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from src.events.state import ObjectState
from src.events.types import Alert, AlertStatus, EventType, SecurityEvent, Severity
from src.events.zone import Zone, point_in_zone


TARGET_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


class EventEngine:
    """
    Deterministic surveillance event intelligence engine.
    Interprets unified object state observations against zone geometry, identity context,
    and temporal rules to generate deduplicated security events and alerts.
    """

    def __init__(
        self,
        zones: Optional[List[Zone]] = None,
        loitering_duration: float = 30.0,
        stationary_duration: float = 60.0,
        movement_threshold: float = 15.0,
    ):
        """
        Initializes the event engine.

        Args:
            zones: List of configured Zone objects.
            loitering_duration: Dwell time in seconds to trigger LOITERING for persons in restricted zones.
            stationary_duration: Dwell time in seconds to trigger SUSPICIOUS_VEHICLE for stopped vehicles.
            movement_threshold: Maximum displacement in pixels to consider an object stationary.
        """
        self.zones: List[Zone] = zones if zones is not None else []
        self.loitering_duration: float = float(loitering_duration)
        self.stationary_duration: float = float(stationary_duration)
        self.movement_threshold: float = float(movement_threshold)

        # Track spatial and temporal state
        # (track_id, zone_id) -> entry timestamp
        self._track_zone_entry: Dict[Tuple[int, str], float] = {}

        # track_id -> list of (timestamp, position)
        self._track_positions: Dict[int, List[Tuple[float, Tuple[int, int]]]] = {}

        # track_id -> timestamp when stationary condition began
        self._stationary_since: Dict[int, float] = {}

        # Deduplication tracker: (track_id, event_type_str, zone_id)
        self._triggered_events: Set[Tuple[int, str, str]] = set()

        # In-memory history
        self.active_alerts: List[Alert] = []
        self.event_history: List[SecurityEvent] = []

        # Counter for deterministic identifiers
        self._event_count: int = 0
        self._alert_count: int = 0

    def add_zone(self, zone: Zone) -> None:
        """Adds a surveillance zone to the engine."""
        self.zones.append(zone)

    def set_zones(self, zones: List[Zone]) -> None:
        """Replaces current surveillance zones."""
        self.zones = list(zones)

    def reset(self) -> None:
        """Clears all internal temporal and tracking states."""
        self._track_zone_entry.clear()
        self._track_positions.clear()
        self._stationary_since.clear()
        self._triggered_events.clear()
        self.active_alerts.clear()
        self.event_history.clear()
        self._event_count = 0
        self._alert_count = 0

    def update(
        self,
        object_states: List[ObjectState],
        timestamp: float,
    ) -> Tuple[List[SecurityEvent], List[Alert]]:
        """
        Evaluates active object states at the given timestamp.
        Generates security events and user-facing alerts.

        Args:
            object_states: List of current ObjectState instances from the perception layer.
            timestamp: Current video stream or wall-clock timestamp in seconds.

        Returns:
            Tuple of (new_security_events, new_alerts).
        """
        new_events: List[SecurityEvent] = []
        new_alerts: List[Alert] = []

        active_track_ids = {obj.track_id for obj in object_states}

        # 1. Evaluate each object against all zones
        for obj in object_states:
            tid = obj.track_id
            pos = obj.position

            # Update position trajectory history (retain last 15 seconds)
            if tid not in self._track_positions:
                self._track_positions[tid] = []
            self._track_positions[tid].append((timestamp, pos))
            self._track_positions[tid] = [
                (t, p) for (t, p) in self._track_positions[tid] if timestamp - t <= 15.0
            ]

            # Track primary zone for ObjectState reporting
            primary_zone: Optional[str] = None

            for zone in self.zones:
                is_inside = point_in_zone(pos, zone)
                entry_key = (tid, zone.id)
                was_inside = entry_key in self._track_zone_entry

                if is_inside:
                    primary_zone = zone.id

                # Transition 1: OUTSIDE -> INSIDE (Entry)
                if is_inside and not was_inside:
                    self._track_zone_entry[entry_key] = timestamp

                    # Initialize stationary tracker on entry for vehicles
                    if obj.object_type in TARGET_VEHICLE_CLASSES:
                        self._stationary_since[tid] = timestamp

                    # Evaluate Intrusion Rules
                    if zone.zone_type == "restricted":
                        # Check Unknown Person Intrusion rule vs Generic Intrusion
                        is_unknown_person = (
                            obj.object_type == "person"
                            and obj.has_face_detected
                            and (obj.identity == "UNKNOWN" or obj.identity is None)
                        )

                        if is_unknown_person:
                            # Rule: UNKNOWN_PERSON_INTRUSION
                            evt_key = (tid, EventType.UNKNOWN_PERSON_INTRUSION.value, zone.id)
                            if evt_key not in self._triggered_events:
                                self._triggered_events.add(evt_key)
                                evt, alr = self._create_event_and_alert(
                                    event_type=EventType.UNKNOWN_PERSON_INTRUSION,
                                    severity=Severity.HIGH,
                                    camera_id=obj.camera_id,
                                    track_id=tid,
                                    timestamp=timestamp,
                                    zone_id=zone.id,
                                    zone_name=zone.name,
                                    object_type=obj.object_type,
                                    identity="UNKNOWN",
                                    face_similarity=obj.face_similarity,
                                    bbox=obj.bbox,
                                )
                                new_events.append(evt)
                                new_alerts.append(alr)

                        # Generic INTRUSION rule (fires on any person or vehicle entering restricted zone)
                        evt_key = (tid, EventType.INTRUSION.value, zone.id)
                        if evt_key not in self._triggered_events:
                            self._triggered_events.add(evt_key)
                            id_label = obj.identity if obj.has_face_detected else "UNVERIFIED"
                            evt, alr = self._create_event_and_alert(
                                event_type=EventType.INTRUSION,
                                severity=Severity.HIGH,
                                camera_id=obj.camera_id,
                                track_id=tid,
                                timestamp=timestamp,
                                zone_id=zone.id,
                                zone_name=zone.name,
                                object_type=obj.object_type,
                                identity=id_label,
                                face_similarity=obj.face_similarity,
                                plate=obj.plate,
                                plate_confidence=obj.plate_confidence,
                                bbox=obj.bbox,
                            )
                            new_events.append(evt)
                            new_alerts.append(alr)

                # Transition 2: INSIDE -> OUTSIDE (Exit)
                elif not is_inside and was_inside:
                    # Clear zone dwell timer
                    self._track_zone_entry.pop(entry_key, None)

                    # Clear triggered event suppression for this track and zone
                    self._triggered_events.discard((tid, EventType.INTRUSION.value, zone.id))
                    self._triggered_events.discard((tid, EventType.UNKNOWN_PERSON_INTRUSION.value, zone.id))
                    self._triggered_events.discard((tid, EventType.LOITERING.value, zone.id))
                    self._triggered_events.discard((tid, EventType.SUSPICIOUS_VEHICLE.value, zone.id))

                    # Reset stationary tracker if exiting vehicle
                    if obj.object_type in TARGET_VEHICLE_CLASSES:
                        self._stationary_since.pop(tid, None)

                # State 3: INSIDE -> INSIDE (Continuous Dwell)
                elif is_inside and was_inside:
                    entry_time = self._track_zone_entry[entry_key]
                    dwell_duration = timestamp - entry_time

                    # Rule: LOITERING (Persons in restricted zone >= loitering_duration)
                    if obj.object_type == "person" and zone.zone_type == "restricted":
                        if dwell_duration >= self.loitering_duration:
                            evt_key = (tid, EventType.LOITERING.value, zone.id)
                            if evt_key not in self._triggered_events:
                                self._triggered_events.add(evt_key)
                                evt, alr = self._create_event_and_alert(
                                    event_type=EventType.LOITERING,
                                    severity=Severity.MEDIUM,
                                    camera_id=obj.camera_id,
                                    track_id=tid,
                                    timestamp=timestamp,
                                    zone_id=zone.id,
                                    zone_name=zone.name,
                                    object_type=obj.object_type,
                                    identity=obj.identity,
                                    face_similarity=obj.face_similarity,
                                    duration=dwell_duration,
                                    bbox=obj.bbox,
                                )
                                new_events.append(evt)
                                new_alerts.append(alr)

                    # Rule: SUSPICIOUS_VEHICLE (Vehicles in restricted/warning zone stationary >= stationary_duration)
                    if obj.object_type in TARGET_VEHICLE_CLASSES and zone.zone_type in {"restricted", "warning"}:
                        is_stationary = self._is_object_stationary(tid)
                        if is_stationary:
                            if tid not in self._stationary_since:
                                self._stationary_since[tid] = timestamp
                            stationary_time = timestamp - self._stationary_since[tid]

                            if stationary_time >= self.stationary_duration:
                                evt_key = (tid, EventType.SUSPICIOUS_VEHICLE.value, zone.id)
                                if evt_key not in self._triggered_events:
                                    self._triggered_events.add(evt_key)
                                    evt, alr = self._create_event_and_alert(
                                        event_type=EventType.SUSPICIOUS_VEHICLE,
                                        severity=Severity.MEDIUM,
                                        camera_id=obj.camera_id,
                                        track_id=tid,
                                        timestamp=timestamp,
                                        zone_id=zone.id,
                                        zone_name=zone.name,
                                        object_type=obj.object_type,
                                        plate=obj.plate,
                                        plate_confidence=obj.plate_confidence,
                                        duration=stationary_time,
                                        bbox=obj.bbox,
                                    )
                                    new_events.append(evt)
                                    new_alerts.append(alr)
                        else:
                            # Moving vehicle resets stationary timer
                            self._stationary_since[tid] = timestamp

            # Update object's zone fields
            obj.previous_zone = obj.current_zone
            obj.current_zone = primary_zone

        # 2. Cleanup terminated/lost tracks
        self._cleanup_stale_tracks(active_track_ids)

        # 3. Store in history
        self.event_history.extend(new_events)
        self.active_alerts.extend(new_alerts)

        return new_events, new_alerts

    def _is_object_stationary(self, track_id: int) -> bool:
        """Determines if a tracked object's displacement is below movement_threshold."""
        history = self._track_positions.get(track_id, [])
        if len(history) < 2:
            return True

        # Calculate max displacement from the earliest position in the window
        init_pos = history[0][1]
        max_disp = 0.0
        for _, pos in history[1:]:
            dx = pos[0] - init_pos[0]
            dy = pos[1] - init_pos[1]
            disp = math.sqrt(dx * dx + dy * dy)
            if disp > max_disp:
                max_disp = disp

        return max_disp < self.movement_threshold

    def _cleanup_stale_tracks(self, active_track_ids: Set[int]) -> None:
        """Cleans up internal state for tracks that are no longer active."""
        # Find entries belonging to inactive tracks
        stale_entries = [k for k in self._track_zone_entry if k[0] not in active_track_ids]
        for k in stale_entries:
            self._track_zone_entry.pop(k, None)

        stale_positions = [tid for tid in self._track_positions if tid not in active_track_ids]
        for tid in stale_positions:
            self._track_positions.pop(tid, None)

        stale_stationary = [tid for tid in self._stationary_since if tid not in active_track_ids]
        for tid in stale_stationary:
            self._stationary_since.pop(tid, None)

        # Clean triggered events for deleted tracks
        stale_events = {k for k in self._triggered_events if k[0] not in active_track_ids}
        self._triggered_events.difference_update(stale_events)

    def _create_event_and_alert(
        self,
        event_type: EventType,
        severity: Severity,
        camera_id: str,
        track_id: int,
        timestamp: float,
        zone_id: str,
        zone_name: str,
        object_type: str,
        identity: Optional[str] = None,
        face_similarity: Optional[float] = None,
        plate: Optional[str] = None,
        plate_confidence: Optional[float] = None,
        duration: Optional[float] = None,
        bbox: Optional[Any] = None,
    ) -> Tuple[SecurityEvent, Alert]:
        """Builds a SecurityEvent and its corresponding operational Alert with human-readable text."""
        self._event_count += 1
        self._alert_count += 1

        event_id = f"evt_{self._event_count:06d}_{uuid.uuid4().hex[:6]}"
        alert_id = f"alr_{self._alert_count:06d}_{uuid.uuid4().hex[:6]}"

        metadata: Dict[str, Any] = {
            "object_type": object_type,
            "zone_name": zone_name,
        }
        if identity is not None:
            metadata["identity"] = identity
        if face_similarity is not None:
            metadata["face_similarity"] = face_similarity
        if plate is not None:
            metadata["plate"] = plate
        if plate_confidence is not None:
            metadata["plate_confidence"] = plate_confidence
        if duration is not None:
            metadata["duration"] = duration
        if bbox is not None:
            metadata["bbox"] = bbox.as_tuple() if hasattr(bbox, "as_tuple") else bbox

        # Human-readable message generation
        if event_type == EventType.UNKNOWN_PERSON_INTRUSION:
            title = "🚨 UNKNOWN PERSON INTRUSION"
            conf_str = f" (Confidence: {face_similarity:.2f})" if face_similarity is not None else ""
            message = (
                f"🚨 UNKNOWN PERSON INTRUSION\n\n"
                f"Camera: {camera_id}\n"
                f"Track: Person #{track_id}\n"
                f"Identity: UNKNOWN{conf_str}\n"
                f"Zone: {zone_name}\n"
                f"Severity: {severity.value}"
            )
        elif event_type == EventType.INTRUSION:
            title = "🚨 INTRUSION DETECTED"
            id_str = f"\nIdentity: {identity}" if identity else ""
            plate_str = f"\nPlate: {plate}" if plate else ""
            message = (
                f"🚨 INTRUSION DETECTED\n\n"
                f"Camera: {camera_id}\n"
                f"Object: {object_type.capitalize()} #{track_id}{id_str}{plate_str}\n"
                f"Zone: {zone_name}\n"
                f"Severity: {severity.value}"
            )
        elif event_type == EventType.LOITERING:
            title = "⚠️ LOITERING DETECTED"
            dur_str = f"{int(duration)}s" if duration is not None else "Extended"
            message = (
                f"⚠️ LOITERING DETECTED\n\n"
                f"Camera: {camera_id}\n"
                f"Object: {object_type.capitalize()} #{track_id}\n"
                f"Zone: {zone_name}\n"
                f"Duration: {dur_str}\n"
                f"Severity: {severity.value}"
            )
        elif event_type == EventType.SUSPICIOUS_VEHICLE:
            title = "⚠️ SUSPICIOUS VEHICLE"
            dur_str = f"{int(duration)}s" if duration is not None else "Extended"
            plate_str = f"\nPlate: {plate}" if plate else ""
            message = (
                f"⚠️ SUSPICIOUS VEHICLE\n\n"
                f"Camera: {camera_id}\n"
                f"Vehicle: {object_type.capitalize()} #{track_id}{plate_str}\n"
                f"Zone: {zone_name}\n"
                f"Stationary Duration: {dur_str}\n"
                f"Severity: {severity.value}"
            )
        else:
            title = f"SECURITY EVENT: {event_type.value}"
            message = f"Event {event_type.value} on Track #{track_id} in {zone_name}"

        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            camera_id=camera_id,
            track_id=track_id,
            timestamp=timestamp,
            zone_id=zone_id,
            message=message,
            metadata=metadata,
        )

        alert = Alert(
            alert_id=alert_id,
            event_id=event_id,
            severity=severity,
            title=title,
            message=message,
            timestamp=timestamp,
            status=AlertStatus.NEW,
            metadata=metadata,
        )

        return event, alert
