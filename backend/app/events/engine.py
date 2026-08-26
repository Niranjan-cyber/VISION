from typing import Dict, Any, List

class RiskEngine:
    """Risk Assessment Engine for processing track trajectories and virtual fence alerts."""

    def __init__(self):
        self.risk_thresholds = {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4
        }

    def evaluate_risk(self, track: Dict[str, Any], is_inside_fence: bool, loiter_time_sec: float) -> str:
        """Determines risk level based on object type, virtual fence intrusion, and loitering duration."""
        if is_inside_fence:
            if loiter_time_sec > 10:
                return "CRITICAL"
            return "HIGH"
        elif loiter_time_sec > 30:
            return "MEDIUM"
        return "LOW"
