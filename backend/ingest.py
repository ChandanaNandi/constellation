"""Ingest sample images into database."""
import asyncio
from pathlib import Path
from app.db import async_session_maker
from app.services.ingestion import ingest_bdd100k

async def run():
    async with async_session_maker() as session:
        stats = await ingest_bdd100k(
            session,
            images_dir=Path('/app/data/bdd100k/images/100k/train'),
            labels_file=Path('/app/data/bdd100k/labels/det_20/det_train.json'),
            max_images=100
        )
        print(stats)

if __name__ == "__main__":
    asyncio.run(run())
