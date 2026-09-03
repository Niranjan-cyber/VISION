from typing import List, Optional
from src.core.types import BoundingBox, PlateRecognitionResult, Track, VehiclePlateAssociation

TARGET_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


def map_crop_to_global_bbox(
    crop_bbox: BoundingBox,
    parent_bbox: BoundingBox,
    frame_w: int,
    frame_h: int,
) -> BoundingBox:
    """
    Translates a bounding box relative to a vehicle crop into absolute video frame coordinates.
    """
    gx1 = max(0, min(parent_bbox.x1 + crop_bbox.x1, frame_w - 1))
    gy1 = max(0, min(parent_bbox.y1 + crop_bbox.y1, frame_h - 1))
    gx2 = max(0, min(parent_bbox.x1 + crop_bbox.x2, frame_w - 1))
    gy2 = max(0, min(parent_bbox.y1 + crop_bbox.y2, frame_h - 1))

    if gx2 <= gx1:
        gx2 = min(frame_w, gx1 + 1)
    if gy2 <= gy1:
        gy2 = min(frame_h, gy1 + 1)

    return BoundingBox(x1=gx1, y1=gy1, x2=gx2, y2=gy2)


def associate_plates_to_vehicles(
    vehicle_tracks: List[Track],
    plates: List[PlateRecognitionResult],
) -> List[VehiclePlateAssociation]:
    """
    Associates detected and recognized plates to the corresponding vehicle track.
    Assumes plate bounding boxes are in global frame coordinates.
    """
    associations: List[VehiclePlateAssociation] = []

    valid_vehicles = [t for t in vehicle_tracks if t.class_name in TARGET_VEHICLE_CLASSES]
    if not valid_vehicles or not plates:
        return associations

    for plate in plates:
        pb = plate.bbox
        best_vehicle: Optional[Track] = None
        best_overlap_score = 0.0

        for veh in valid_vehicles:
            vb = veh.bbox

            # Check if plate center is inside vehicle bounding box
            pc_x, pc_y = pb.center
            if vb.x1 <= pc_x <= vb.x2 and vb.y1 <= pc_y <= vb.y2:
                # Calculate intersection area over plate area (containment ratio)
                ix1 = max(pb.x1, vb.x1)
                iy1 = max(pb.y1, vb.y1)
                ix2 = min(pb.x2, vb.x2)
                iy2 = min(pb.y2, vb.y2)

                inter_w = max(0, ix2 - ix1)
                inter_h = max(0, iy2 - iy1)
                inter_area = inter_w * inter_h

                containment = inter_area / float(pb.area) if pb.area > 0 else 0.0

                if containment > best_overlap_score and containment >= 0.50:
                    best_overlap_score = containment
                    best_vehicle = veh

        if best_vehicle is not None:
            associations.append(
                VehiclePlateAssociation(
                    track_id=best_vehicle.track_id,
                    vehicle_class=best_vehicle.class_name,
                    plate=plate,
                )
            )

    return associations
