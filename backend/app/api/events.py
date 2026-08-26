from fastapi import APIRouter, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class DetectionEvent(BaseModel):
    event_id: str
    camera_id: str
    timestamp: datetime
    track_id: int
    object_type: str
    risk_level: str
    location: List[int]
    zone_id: Optional[str] = None

@router.get("/", response_model=List[DetectionEvent])
async def get_events(
    camera_id: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: int = Query(default=50, le=100)
):
    return []
