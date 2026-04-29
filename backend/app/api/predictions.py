"""Prediction and inference endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db

router = APIRouter()


@router.post("/predict")
async def run_prediction(
    image_id: int,
    model_id: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Run inference on an image."""
    # TODO: Implement in Phase 3
    return {"message": "Prediction endpoint - coming soon"}


@router.get("/disagreements")
async def list_disagreements(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List shadow mode disagreements."""
    # TODO: Implement in Phase 4
    return {"disagreements": [], "total": 0}
