#!/usr/bin/env python3
"""CLI script to ingest BDD100K data into PostgreSQL.

Usage:
    python scripts/ingest_data.py --images-dir data/bdd100k/images/100k/train \
                                   --labels-file data/bdd100k/labels/det_20/det_train.json \
                                   --max-images 5000
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main():
    parser = argparse.ArgumentParser(description="Ingest BDD100K data into database")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/bdd100k/images/100k/train"),
        help="Path to images directory",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=Path("data/bdd100k/labels/det_20/det_train.json"),
        help="Path to labels JSON file",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to ingest (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for database commits (default: 100)",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Database URL (default: from DATABASE_URL env var)",
    )
    args = parser.parse_args()

    # Validate paths
    if not args.images_dir.exists():
        print(f"Error: Images directory not found: {args.images_dir}")
        print("Run scripts/download_bdd100k.sh first to download the dataset.")
        sys.exit(1)

    if not args.labels_file.exists():
        print(f"Warning: Labels file not found: {args.labels_file}")
        print("Images will be ingested without labels.")

    # Import after path setup
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    import os

    from backend.app.services.ingestion import ingest_bdd100k
    from backend.app.db.models import Base

    # Get database URL
    database_url = args.database_url or os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://constellation:constellation_dev@localhost:5432/constellation"
    )

    print(f"Connecting to database: {database_url.split('@')[-1]}")

    # Create engine and session
    engine = create_async_engine(database_url, echo=False)

    # Create tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    # Run ingestion
    async with async_session() as session:
        stats = await ingest_bdd100k(
            session=session,
            images_dir=args.images_dir,
            labels_file=args.labels_file,
            batch_size=args.batch_size,
            max_images=args.max_images,
        )

    print("\nIngestion Results:")
    print(f"  Images processed: {stats['images_processed']}")
    print(f"  Images created: {stats['images_created']}")
    print(f"  Labels created: {stats['labels_created']}")
    if stats['errors']:
        print(f"  Errors: {len(stats['errors'])}")
        for error in stats['errors'][:5]:
            print(f"    - {error}")
        if len(stats['errors']) > 5:
            print(f"    ... and {len(stats['errors']) - 5} more")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
