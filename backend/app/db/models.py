"""Database models for Constellation."""

from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import String, Text, Float, Integer, Enum, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class LabelSource(PyEnum):
    """Source of image labels."""
    MANUAL = "manual"
    AUTO_SAM = "auto_sam"
    AUTO_YOLO = "auto_yolo"
    AUTO_COMBINED = "auto_combined"


class ImageStatus(PyEnum):
    """Processing status of an image."""
    PENDING = "pending"
    PROCESSING = "processing"
    LABELED = "labeled"
    FAILED = "failed"


class Image(Base):
    """Represents a driving scene image in the dataset."""

    __tablename__ = "images"

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[ImageStatus] = mapped_column(
        Enum(ImageStatus), default=ImageStatus.PENDING
    )
    dataset: Mapped[str] = mapped_column(String(50), default="bdd100k")
    split: Mapped[str] = mapped_column(String(20), default="train")  # train/val/test

    # Relationships
    labels: Mapped[list["Label"]] = relationship(back_populates="image", cascade="all, delete-orphan")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="image", cascade="all, delete-orphan")


class Label(Base):
    """Ground truth labels for an image."""

    __tablename__ = "labels"

    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), nullable=False)
    source: Mapped[LabelSource] = mapped_column(Enum(LabelSource), nullable=False)
    task: Mapped[str] = mapped_column(String(50), nullable=False)  # detection, lane, drivable, depth, traffic_light
    data: Mapped[dict] = mapped_column(JSON, nullable=False)  # COCO format or task-specific

    # Relationships
    image: Mapped["Image"] = relationship(back_populates="labels")


class Model(Base):
    """Represents a trained model checkpoint."""

    __tablename__ = "models"

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(String(512), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=True)  # mAP, IoU, etc.
    is_baseline: Mapped[bool] = mapped_column(default=False)
    is_active: Mapped[bool] = mapped_column(default=False)

    # Relationships
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model")


class Prediction(Base):
    """Model predictions on an image (for shadow mode comparison)."""

    __tablename__ = "predictions"

    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), nullable=False)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    task: Mapped[str] = mapped_column(String(50), nullable=False)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=True)

    # Relationships
    image: Mapped["Image"] = relationship(back_populates="predictions")
    model: Mapped["Model"] = relationship(back_populates="predictions")


class Disagreement(Base):
    """Records disagreements between baseline and candidate models."""

    __tablename__ = "disagreements"

    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), nullable=False)
    baseline_model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    candidate_model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    task: Mapped[str] = mapped_column(String(50), nullable=False)
    disagreement_type: Mapped[str] = mapped_column(String(50), nullable=False)  # missing, extra, wrong_class, etc.
    severity: Mapped[float] = mapped_column(Float, nullable=False)  # 0-1 score
    details: Mapped[dict] = mapped_column(JSON, nullable=True)
