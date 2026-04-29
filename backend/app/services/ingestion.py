"""Data ingestion service for populating database from BDD100K.

Takes a folder of BDD100K images + labels JSON, populates Image and Label tables.
"""

import json
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from PIL import Image as PILImage

from app.db.models import Image, Label, LabelSource, ImageStatus


async def ingest_bdd100k(
    session: AsyncSession,
    images_dir: Path,
    labels_file: Path,
    batch_size: int = 100,
    max_images: int | None = None,
) -> dict:
    """Ingest BDD100K images and labels into the database.

    Args:
        session: Async database session
        images_dir: Path to directory containing images
        labels_file: Path to detection labels JSON file
        batch_size: Number of records to commit at once
        max_images: Maximum number of images to ingest (None = all)

    Returns:
        Dictionary with ingestion statistics
    """
    stats = {
        "images_processed": 0,
        "images_created": 0,
        "labels_created": 0,
        "errors": [],
    }

    # Load labels
    if labels_file.exists():
        with open(labels_file, "r") as f:
            labels_data = json.load(f)
        labels_by_name = {
            Path(item["name"]).stem: item.get("labels", [])
            for item in labels_data
        }
    else:
        labels_by_name = {}
        stats["errors"].append(f"Labels file not found: {labels_file}")

    # Get list of images
    image_files = sorted(images_dir.glob("*.jpg"))
    if max_images:
        image_files = image_files[:max_images]

    print(f"Ingesting {len(image_files)} images from {images_dir}")

    # Process images in batches
    batch_images = []
    batch_labels = []

    for i, image_path in enumerate(image_files):
        try:
            # Check if image already exists
            image_name = image_path.stem
            existing = await session.execute(
                select(Image).where(Image.file_path == str(image_path))
            )
            if existing.scalar_one_or_none():
                stats["images_processed"] += 1
                continue

            # Get image dimensions
            with PILImage.open(image_path) as img:
                width, height = img.size

            # Create Image record
            image_record = Image(
                filename=image_path.name,
                file_path=str(image_path),
                width=width,
                height=height,
                status=ImageStatus.LABELED if image_name in labels_by_name else ImageStatus.PENDING,
                dataset="bdd100k",
                split="train",
            )
            batch_images.append(image_record)
            stats["images_created"] += 1

            # Create Label records if we have labels
            if image_name in labels_by_name:
                labels = labels_by_name[image_name]
                if labels:
                    label_record = Label(
                        image=image_record,
                        source=LabelSource.MANUAL,
                        task="detection",
                        data={"labels": labels},
                    )
                    batch_labels.append(label_record)
                    stats["labels_created"] += 1

        except Exception as e:
            stats["errors"].append(f"{image_path.name}: {str(e)}")

        stats["images_processed"] += 1

        # Commit batch
        if len(batch_images) >= batch_size:
            session.add_all(batch_images)
            session.add_all(batch_labels)
            await session.commit()
            batch_images = []
            batch_labels = []
            print(f"  Processed {stats['images_processed']}/{len(image_files)}")

    # Commit remaining
    if batch_images:
        session.add_all(batch_images)
        session.add_all(batch_labels)
        await session.commit()

    print(f"Ingestion complete: {stats['images_created']} images, {stats['labels_created']} labels")
    return stats


async def ingest_auto_label(
    session: AsyncSession,
    image_id: int,
    label_data: dict,
) -> Label:
    """Save auto-generated labels for an image.

    Args:
        session: Database session
        image_id: ID of the image
        label_data: Auto-label results (boxes, masks, etc.)

    Returns:
        Created Label record
    """
    # Update image status
    image = await session.get(Image, image_id)
    if not image:
        raise ValueError(f"Image not found: {image_id}")

    image.status = ImageStatus.LABELED

    # Create label record
    label = Label(
        image_id=image_id,
        source=LabelSource.AUTO_COMBINED,
        task="detection",
        data=label_data,
    )
    session.add(label)
    await session.commit()
    await session.refresh(label)

    return label


async def get_ingestion_stats(session: AsyncSession) -> dict:
    """Get current ingestion statistics."""
    from sqlalchemy import func

    total_images = await session.execute(select(func.count(Image.id)))
    total_labels = await session.execute(select(func.count(Label.id)))

    pending = await session.execute(
        select(func.count(Image.id)).where(Image.status == ImageStatus.PENDING)
    )
    labeled = await session.execute(
        select(func.count(Image.id)).where(Image.status == ImageStatus.LABELED)
    )

    return {
        "total_images": total_images.scalar_one(),
        "total_labels": total_labels.scalar_one(),
        "pending_images": pending.scalar_one(),
        "labeled_images": labeled.scalar_one(),
    }
