from fastapi import APIRouter

from app.api import images, models, predictions

router = APIRouter(prefix="/api")

router.include_router(images.router, prefix="/images", tags=["images"])
router.include_router(models.router, prefix="/models", tags=["models"])
router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
