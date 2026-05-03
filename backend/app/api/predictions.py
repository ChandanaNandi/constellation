"""Prediction and inference endpoints."""

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from PIL import Image
import io

from app.db import get_db
from app.db.models import Image as ImageModel
from app.services.inference import inference_service

router = APIRouter()


class PredictionRequest(BaseModel):
    image_id: int
    model_id: int | None = None
    score_threshold: float = 0.3


class DetectionResult(BaseModel):
    box: list[float]
    class_id: int
    class_name: str
    confidence: float


class SegmentationResult(BaseModel):
    road_percentage: float
    sidewalk_percentage: float


class PredictionResponse(BaseModel):
    image_id: int
    num_detections: int
    detections: list[DetectionResult]
    segmentation: SegmentationResult
    image_size: dict


@router.post("/predict", response_model=PredictionResponse)
async def run_prediction(
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run HydraNet V2 inference on an image from the database.

    Returns detection boxes and segmentation percentages.
    """
    # Get image from database
    image_record = await db.get(ImageModel, request.image_id)
    if not image_record:
        raise HTTPException(status_code=404, detail="Image not found")

    # Check if file exists
    file_path = Path(image_record.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    try:
        # Load image
        pil_image = Image.open(file_path).convert('RGB')

        # Run inference
        results = inference_service.predict(
            pil_image,
            score_threshold=request.score_threshold,
        )

        return PredictionResponse(
            image_id=request.image_id,
            num_detections=results['num_detections'],
            detections=[DetectionResult(**d) for d in results['detections']],
            segmentation=SegmentationResult(**results['segmentation']),
            image_size=results['image_size'],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/predict-upload")
async def predict_from_upload(
    file: UploadFile = File(...),
    score_threshold: float = Query(default=0.3, ge=0.1, le=0.9),
):
    """Run inference on an uploaded image (no database required).

    This endpoint accepts a direct image upload and returns predictions.
    Useful for quick testing without storing images.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image from upload
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Run inference
        results = inference_service.predict(
            pil_image,
            score_threshold=score_threshold,
        )

        return {
            "filename": file.filename,
            "num_detections": results['num_detections'],
            "detections": results['detections'],
            "segmentation": results['segmentation'],
            "image_size": results['image_size'],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.get("/disagreements")
async def list_disagreements(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List shadow mode disagreements."""
    # TODO: Implement shadow mode comparison
    return {"disagreements": [], "total": 0}


@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    return {
        "model_name": "HydraNet V2",
        "architecture": {
            "backbone": "EfficientNet-B0",
            "detection_head": "FCOS (anchor-free)",
            "segmentation_head": "Dilated convolutions",
        },
        "detection_classes": [
            "person", "rider", "car", "truck",
            "bus", "train", "motorcycle", "bicycle"
        ],
        "segmentation_classes": ["background", "road", "sidewalk"],
        "input_size": {"height": 512, "width": 1024},
        "trained_on": "Cityscapes",
        "metrics": {
            "segmentation_iou": "84.9%",
            "training_epochs": 15,
        },
    }
