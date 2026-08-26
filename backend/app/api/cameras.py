from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class CameraConfig(BaseModel):
    id: str
    name: str
    rtsp_url: str
    zone_id: str
    enabled: bool = True

app_cameras = {}

@router.get("/", response_model=List[CameraConfig])
async def list_cameras():
    return list(app_cameras.values())

@router.post("/", response_model=CameraConfig)
async def add_camera(camera: CameraConfig):
    if camera.id in app_cameras:
        raise HTTPException(status_code=400, detail="Camera ID already exists")
    app_cameras[camera.id] = camera
    return camera

@router.get("/{camera_id}", response_model=CameraConfig)
async def get_camera(camera_id: str):
    if camera_id not in app_cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    return app_cameras[camera_id]
