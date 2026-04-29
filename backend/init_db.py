"""Initialize database tables."""
import asyncio
from app.db.models import Base
from app.db import engine

async def init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created!")

if __name__ == "__main__":
    asyncio.run(init())
