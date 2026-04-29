"""Image management and auto-labeling endpoints."""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel
from pathlib import Path

from app.db import get_db
from app.db.models import Image, Label, LabelSource, ImageStatus

router = APIRouter()


class ImageResponse(BaseModel):
    """Response model for a single image."""
    id: int
    filename: str
    file_path: str
    width: int | None
    height: int | None
    status: str
    dataset: str | None
    split: str | None
    created_at: str

    class Config:
        from_attributes = True


class LabelResponse(BaseModel):
    """Response model for a label."""
    id: int
    source: str
    task: str
    data: dict
    created_at: str

    class Config:
        from_attributes = True


class ImageDetailResponse(BaseModel):
    """Response model for image with labels."""
    image: ImageResponse
    labels: list[LabelResponse]


class ImageListResponse(BaseModel):
    """Response model for paginated image list."""
    images: list[ImageResponse]
    total: int
    limit: int
    offset: int


class AutoLabelRequest(BaseModel):
    """Request model for auto-labeling."""
    use_sam: bool = True
    conf_threshold: float = 0.25


class AutoLabelResponse(BaseModel):
    """Response model for auto-label results."""
    image_id: int
    label_id: int
    num_detections: int
    detections: list[dict]


@router.get("", response_model=ImageListResponse)
async def list_images(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None),
    dataset: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """List all images in the dataset with pagination.

    Args:
        limit: Number of images to return (1-100)
        offset: Number of images to skip
        status: Filter by status (pending, labeled, reviewed)
        dataset: Filter by dataset name
    """
    # Build query
    query = select(Image)
    count_query = select(func.count(Image.id))

    # Apply filters
    if status:
        try:
            status_enum = ImageStatus(status)
            query = query.where(Image.status == status_enum)
            count_query = count_query.where(Image.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {[s.value for s in ImageStatus]}"
            )

    if dataset:
        query = query.where(Image.dataset == dataset)
        count_query = count_query.where(Image.dataset == dataset)

    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    # Get paginated results
    query = query.order_by(Image.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    images = result.scalars().all()

    return ImageListResponse(
        images=[
            ImageResponse(
                id=img.id,
                filename=img.filename,
                file_path=img.file_path,
                width=img.width,
                height=img.height,
                status=img.status.value,
                dataset=img.dataset,
                split=img.split,
                created_at=img.created_at.isoformat(),
            )
            for img in images
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{image_id}", response_model=ImageDetailResponse)
async def get_image(
    image_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get details for a specific image including all labels."""
    # Get image
    image = await db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Get all labels for this image
    labels_query = select(Label).where(Label.image_id == image_id).order_by(Label.created_at.desc())
    labels_result = await db.execute(labels_query)
    labels = labels_result.scalars().all()

    return ImageDetailResponse(
        image=ImageResponse(
            id=image.id,
            filename=image.filename,
            file_path=image.file_path,
            width=image.width,
            height=image.height,
            status=image.status.value,
            dataset=image.dataset,
            split=image.split,
            created_at=image.created_at.isoformat(),
        ),
        labels=[
            LabelResponse(
                id=label.id,
                source=label.source.value,
                task=label.task,
                data=label.data,
                created_at=label.created_at.isoformat(),
            )
            for label in labels
        ],
    )


@router.get("/{image_id}/file")
async def get_image_file(
    image_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Serve the actual image file."""
    image = await db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = Path(image.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    return FileResponse(
        path=file_path,
        media_type="image/jpeg",
        filename=image.filename,
    )


@router.post("/{image_id}/auto-label", response_model=AutoLabelResponse)
async def auto_label_image(
    image_id: int,
    request: AutoLabelRequest = AutoLabelRequest(),
    db: AsyncSession = Depends(get_db),
):
    """Run auto-labeling on an image using YOLOv8 + SAM.

    This endpoint triggers the auto-labeling pipeline which:
    1. Runs YOLOv8 object detection
    2. Optionally runs SAM segmentation on detected boxes
    3. Saves results to the Label table
    4. Updates image status to LABELED

    Args:
        image_id: ID of the image to label
        request: Auto-labeling configuration (use_sam, conf_threshold)
    """
    # Get image from database
    image = await db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Check if image file exists
    file_path = Path(image.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    try:
        # Import auto-labeler (lazy load to avoid loading models at startup)
        from data_engine.auto_labeler import AutoLabeler

        # Initialize auto-labeler
        labeler = AutoLabeler(
            use_sam=request.use_sam,
            conf_threshold=request.conf_threshold,
        )

        # Run auto-labeling
        results = labeler.label_image(file_path)

        # Convert masks to serializable format (RLE)
        serializable_results = {
            "boxes": results["boxes"],
            "confidences": results["confidences"],
            "class_ids": results["class_ids"],
            "class_names": results["class_names"],
            "image_shape": list(results["image_shape"]),
        }

        # Add COCO-formatted results
        if results["boxes"]:
            coco_results = labeler.to_coco_format(results, image_id)
            serializable_results["coco_annotations"] = coco_results["annotations"]

        # Save label to database
        label = Label(
            image_id=image_id,
            source=LabelSource.AUTO_COMBINED,
            task="detection",
            data=serializable_results,
        )
        db.add(label)

        # Update image status
        image.status = ImageStatus.LABELED

        await db.commit()
        await db.refresh(label)

        # Build detection summaries for response
        detections = [
            {
                "class_name": cls_name,
                "confidence": conf,
                "bbox": box,
            }
            for cls_name, conf, box in zip(
                results["class_names"],
                results["confidences"],
                results["boxes"],
            )
        ]

        return AutoLabelResponse(
            image_id=image_id,
            label_id=label.id,
            num_detections=len(results["boxes"]),
            detections=detections,
        )

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Auto-labeling dependencies not installed: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Auto-labeling failed: {str(e)}"
        )


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    dataset: str = Query(default="uploaded"),
    db: AsyncSession = Depends(get_db),
):
    """Upload a new image for processing.

    Saves the image to disk and creates a database record.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Create upload directory
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    import uuid
    ext = Path(file.filename or "image.jpg").suffix or ".jpg"
    unique_filename = f"{uuid.uuid4()}{ext}"
    file_path = upload_dir / unique_filename

    # Save file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Get image dimensions
    try:
        from PIL import Image as PILImage
        with PILImage.open(file_path) as img:
            width, height = img.size
    except Exception:
        width, height = None, None

    # Create database record
    image = Image(
        filename=file.filename or unique_filename,
        file_path=str(file_path.absolute()),
        width=width,
        height=height,
        status=ImageStatus.PENDING,
        dataset=dataset,
    )
    db.add(image)
    await db.commit()
    await db.refresh(image)

    return {
        "id": image.id,
        "filename": image.filename,
        "file_path": image.file_path,
        "width": width,
        "height": height,
        "message": "Image uploaded successfully",
    }
