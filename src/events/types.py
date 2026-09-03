from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EventType(str, Enum):
    """Supported security event types in VISION."""

    INTRUSION = "INTRUSION"
    UNKNOWN_PERSON_INTRUSION = "UNKNOWN_PERSON_INTRUSION"
    LOITERING = "LOITERING"
    SUSPICIOUS_VEHICLE = "SUSPICIOUS_VEHICLE"


class Severity(str, Enum):
    """Severity levels for surveillance events and alerts."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertStatus(str, Enum):
    """Lifecycle status of security alerts."""

    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


@dataclass
class SecurityEvent:
    """
    Represents an evaluated surveillance security event.
    Produced deterministically by the EventEngine from unified object state transitions.
    """

    event_id: str
    event_type: EventType
    severity: Severity
    camera_id: str
    track_id: int
    timestamp: float
    zone_id: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """
    User-facing operational alert derived from a SecurityEvent.
    Maintains status lifecycle (NEW, ACKNOWLEDGED, RESOLVED).
    """

    alert_id: str
    event_id: str
    severity: Severity
    title: str
    message: str
    timestamp: float
    status: AlertStatus = AlertStatus.NEW
    metadata: Dict[str, Any] = field(default_factory=dict)
