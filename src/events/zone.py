from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import yaml


VALID_ZONE_TYPES = {"restricted", "warning", "monitored"}


@dataclass
class Zone:
    """Represents a spatial surveillance zone defined by a 2D polygon."""

    id: str
    name: str
    zone_type: str
    polygon: List[Tuple[int, int]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate zone ID
        if not self.id or not isinstance(self.id, str) or not self.id.strip():
            raise ValueError(f"Zone ID must be a non-empty string, got '{self.id}'")
        self.id = self.id.strip()

        # Validate zone name
        if not self.name or not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"Zone name must be a non-empty string, got '{self.name}'")
        self.name = self.name.strip()

        # Validate zone type
        if not self.zone_type or self.zone_type.lower() not in VALID_ZONE_TYPES:
            raise ValueError(
                f"Invalid zone type '{self.zone_type}'. Expected one of {sorted(list(VALID_ZONE_TYPES))}"
            )
        self.zone_type = self.zone_type.lower()

        # Validate polygon
        if not self.polygon or not isinstance(self.polygon, (list, tuple)):
            raise ValueError(f"Zone '{self.id}' polygon must be a non-empty list of points")

        if len(self.polygon) < 3:
            raise ValueError(
                f"Zone '{self.id}' polygon must contain at least 3 vertices, got {len(self.polygon)}"
            )

        validated_poly: List[Tuple[int, int]] = []
        for i, pt in enumerate(self.polygon):
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                raise ValueError(
                    f"Zone '{self.id}' vertex #{i} must be a 2-element coordinate (x, y), got {pt}"
                )
            try:
                x, y = int(pt[0]), int(pt[1])
            except (ValueError, TypeError):
                raise ValueError(
                    f"Zone '{self.id}' vertex #{i} coordinates must be numeric, got {pt}"
                )
            validated_poly.append((x, y))

        self.polygon = validated_poly


def point_in_zone(point: Tuple[int, int], zone: Zone) -> bool:
    """
    Evaluates whether a 2D point is inside or on the boundary of a zone polygon.
    Uses OpenCV's pointPolygonTest for robust geometric evaluation.

    Args:
        point: Tuple (x, y) coordinates of the test point (e.g. object bottom-center).
        zone: Zone instance with polygon vertices.

    Returns:
        True if the point is strictly inside or on the boundary of the polygon, False otherwise.
    """
    if not zone.polygon or len(zone.polygon) < 3:
        return False

    pts_arr = np.array(zone.polygon, dtype=np.int32)
    # measureDist=False returns: +1 (inside), 0 (on edge), -1 (outside)
    dist = cv2.pointPolygonTest(pts_arr, (float(point[0]), float(point[1])), measureDist=False)
    return dist >= 0


def load_zones_from_dict(data: Dict[str, Any]) -> List[Zone]:
    """
    Parses and validates a list of Zone objects from a dictionary structure.

    Args:
        data: Dict containing a 'zones' list.

    Returns:
        List of validated Zone instances.
    """
    if not isinstance(data, dict):
        raise ValueError("Zone configuration root must be a dictionary")

    if "zones" not in data:
        raise ValueError("Zone configuration dictionary missing required 'zones' key")

    raw_zones = data["zones"]
    if not isinstance(raw_zones, list) or len(raw_zones) == 0:
        raise ValueError("Zone configuration 'zones' must be a non-empty list of zone definitions")

    zones: List[Zone] = []
    seen_ids = set()

    for idx, raw_z in enumerate(raw_zones):
        if not isinstance(raw_z, dict):
            raise ValueError(f"Zone definition #{idx} must be a dictionary")

        zone_id = raw_z.get("id")
        if zone_id in seen_ids:
            raise ValueError(f"Duplicate zone ID detected: '{zone_id}'")
        seen_ids.add(zone_id)

        zone = Zone(
            id=zone_id,
            name=raw_z.get("name", zone_id),
            zone_type=raw_z.get("type", "restricted"),
            polygon=raw_z.get("polygon", []),
            metadata=raw_z.get("metadata", {}),
        )
        zones.append(zone)

    return zones


def load_zones_from_file(config_path: str) -> List[Zone]:
    """
    Loads and validates zone definitions from a YAML configuration file.

    Args:
        config_path: Path to the YAML zones configuration file.

    Returns:
        List of validated Zone instances.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Zone configuration file not found at '{config_path}'")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return load_zones_from_dict(data)
