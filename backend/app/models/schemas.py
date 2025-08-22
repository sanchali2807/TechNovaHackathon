from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class LocationMetadata(BaseModel):
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    address: str | None = None
    city: str | None = None
    state: str | None = None
    pincode: str | None = None


class UploadMetadata(BaseModel):
    timestamp: datetime
    location: LocationMetadata | None = None

    # Optional contextual signals for placement checks (if client can supply)
    distance_to_junction_m: float | None = Field(default=None, ge=0)
    distance_to_traffic_light_m: float | None = Field(default=None, ge=0)
    is_restricted_zone: bool | None = None


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    confidence: float | None = None


class DetectionFeatures(BaseModel):
    billboard_count: int
    estimated_area_sqft: float | None = None
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    qr_or_license_present: bool | None = None
    text_content: list[str] | None = None


class ComplianceCheck(BaseModel):
    id: str
    type: Literal["size", "placement", "content", "license"]
    name: str
    status: Literal["violation", "compliant"]
    passed: bool
    details: str
    policy_reference: str | None = None
    references: list[str] = Field(default_factory=list)


class ComplianceReport(BaseModel):
    overall_passed: bool
    checks: list[ComplianceCheck]


class StoredFileInfo(BaseModel):
    stored_filename: str
    local_path: str
    public_url: str | None = None
    is_video: bool = False


class AnalysisResponse(BaseModel):
    file_id: str
    filename: str
    storage_url: str | None
    media_type: Literal["image", "video"]
    detections: DetectionFeatures
    compliance: ComplianceReport
    status: Literal["success", "no_billboard", "uncertain"] = "success"
    message: str | None = None


