"""Prediction and inference endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db

router = APIRouter()


class PredictionRequest(BaseModel):
    image_id: int
    model_id: int | None = None


@router.post("/predict")
async def run_prediction(
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run inference on an image."""
    # TODO: Implement in Phase 3
    return {
        "message": "Prediction endpoint - coming soon",
        "image_id": request.image_id,
        "model_id": request.model_id,
    }


@router.get("/disagreements")
async def list_disagreements(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List shadow mode disagreements."""
    # TODO: Implement in Phase 4
    return {"disagreements": [], "total": 0}
