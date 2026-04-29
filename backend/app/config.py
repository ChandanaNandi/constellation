from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://constellation:constellation_dev@localhost:5432/constellation"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Application
    debug: bool = False
    app_name: str = "Constellation"
    app_version: str = "0.1.0"

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Weights & Biases
    wandb_api_key: str | None = None
    wandb_project: str = "constellation"

    # Model paths
    model_checkpoint_dir: str = "./checkpoints"
    data_dir: str = "./data"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
