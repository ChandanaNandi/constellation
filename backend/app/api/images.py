"""Image upload and management endpoints."""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db

router = APIRouter()


@router.get("")
async def list_images(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List all images in the dataset."""
    # TODO: Implement in Phase 1
    return {"images": [], "total": 0, "limit": limit, "offset": offset}


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a new image for processing."""
    # TODO: Implement in Phase 1
    return {"message": "Upload endpoint - coming soon", "filename": file.filename}


@router.get("/{image_id}")
async def get_image(
    image_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get details for a specific image."""
    # TODO: Implement in Phase 1
    raise HTTPException(status_code=404, detail="Image not found")
