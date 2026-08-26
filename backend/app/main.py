from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import cameras, events, alerts

app = FastAPI(
    title="VISION - Border Surveillance API",
    description="AI-powered Real-Time Video Analytics & Border Surveillance Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cameras.router, prefix="/api/v1/cameras", tags=["Cameras"])
app.include_router(events.router, prefix="/api/v1/events", tags=["Events"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["Alerts"])

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "service": "VISION Backend API"}
