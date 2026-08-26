import os
import redis.asyncio as redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class RedisStorage:
    """Fast temporary tracking data & active trajectory caching."""

    def __init__(self):
        self.client = None

    async def connect(self):
        self.client = redis.from_url(REDIS_URL, decode_responses=True)

    async def save_track_point(self, track_id: int, timestamp: str, pos_x: int, pos_y: int):
        if self.client:
            key = f"track:{track_id}"
            value = f"{timestamp},{pos_x},{pos_y}"
            await self.client.rpush(key, value)
            await self.client.expire(key, 3600)

    async def get_trajectory(self, track_id: int):
        if self.client:
            return await self.client.lrange(f"track:{track_id}", 0, -1)
        return []
