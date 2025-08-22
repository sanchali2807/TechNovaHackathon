from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="BILLBOARD_GUARD_", case_sensitive=False)

    # Storage
    storage_root: Path = Path("backend/uploads").absolute()
    public_base_url: str | None = None  # Optional public URL base if behind CDN/proxy

    # AI model provider: mock | ultralytics | detectron2 (mock by default)
    model_provider: str = "mock"

    # Municipal limits (example: 40 sq ft)
    municipal_area_limit_sqft: float = 40.0

    # Policies path (project root by default)
    policies_path: Path = Path("policies.json").absolute()

    # MVP detection thresholds
    presence_threshold_low: float = 0.3
    presence_threshold_high: float = 0.5
    detection_conf_threshold: float = 0.3
    presence_enabled: bool = False


settings = Settings()

# Ensure storage directory exists early
settings.storage_root.mkdir(parents=True, exist_ok=True)


