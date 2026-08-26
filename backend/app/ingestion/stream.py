import asyncio
from typing import Optional
import numpy as np

class StreamQueue:
    """Async frame buffer queue manager for concurrent video processing."""

    def __init__(self, maxsize: int = 30):
        self.queue = asyncio.Queue(maxsize=maxsize)

    async def push_frame(self, frame: np.ndarray):
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self.queue.put(frame)

    async def pop_frame(self) -> Optional[np.ndarray]:
        if not self.queue.empty():
            return await self.queue.get()
        return None
