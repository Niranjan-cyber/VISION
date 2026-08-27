from dataclasses import dataclass
from typing import List, Optional

from src.core.types import BoundingBox, FaceDetection, Track


@dataclass
class FaceTrackAssociation:
    """Associates a detected FaceDetection with a persistent Track ID."""

    track_id: int
    face: FaceDetection


def calculate_iou(boxA: BoundingBox, boxB: BoundingBox) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA.x1, boxB.x1)
    yA = max(boxA.y1, boxB.y1)
    xB = min(boxA.x2, boxB.x2)
    yB = min(boxA.y2, boxB.y2)

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    if interArea == 0:
        return 0.0

    boxAArea = boxA.area
    boxBArea = boxB.area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def associate_faces_to_tracks(
    tracks: List[Track],
    face_detections: List[FaceDetection],
) -> List[FaceTrackAssociation]:
    """
    Deterministically associates full-frame FaceDetection objects with active person Tracks
    based purely on spatial containment and bounding box geometry.
    """
    if not tracks or not face_detections:
        return []

    # Strict Scope: ONLY person tracks can receive face associations
    person_tracks = [t for t in tracks if t.class_name == "person"]
    if not person_tracks:
        return []

    associations: List[FaceTrackAssociation] = []

    for face in face_detections:
        fc_x, fc_y = face.bbox.center

        # Find candidates whose bounding box spatially contains the face center
        candidates = [
            trk
            for trk in person_tracks
            if (
                trk.bbox.x1 <= fc_x <= trk.bbox.x2
                and trk.bbox.y1 <= fc_y <= trk.bbox.y2
            )
        ]

        if not candidates:
            # Face center not inside any tracked person box -> unassociated
            continue

        if len(candidates) == 1:
            best_person = candidates[0]
        else:
            # Disambiguate multiple candidates using highest IoU
            best_person = max(
                candidates,
                key=lambda p: calculate_iou(face.bbox, p.bbox),
            )

        associations.append(
            FaceTrackAssociation(track_id=best_person.track_id, face=face)
        )

    return associations
