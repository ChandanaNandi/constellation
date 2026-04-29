"""Model management endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db

router = APIRouter()


@router.get("")
async def list_models(db: AsyncSession = Depends(get_db)):
    """List all registered models."""
    # TODO: Implement in Phase 2
    return {"models": []}


@router.get("/{model_id}")
async def get_model(model_id: int, db: AsyncSession = Depends(get_db)):
    """Get details for a specific model."""
    # TODO: Implement in Phase 2
    return {"message": "Model endpoint - coming soon"}
