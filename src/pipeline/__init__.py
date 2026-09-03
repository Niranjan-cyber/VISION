"""VISION reusable pipeline session — shared by the CLI (src/main.py) and the FastAPI backend."""

from src.pipeline.session import FrameResult, PipelineSession, PipelineSubsystemError

__all__ = ["PipelineSession", "FrameResult", "PipelineSubsystemError"]
