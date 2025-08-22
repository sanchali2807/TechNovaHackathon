from __future__ import annotations

import os
import uuid
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .config import settings
from .models.schemas import (
    AnalysisResponse,
    UploadMetadata,
    DetectionFeatures,
    ComplianceReport,
)
from .services.storage import save_upload_file
from .services.compliance import run_compliance_checks
from .pipeline.mock import analyze_media as analyze_media_mock
try:
    from .pipeline.ultralytics import analyze_media as analyze_media_ultra
except Exception:  # noqa: BLE001
    analyze_media_ultra = None  # type: ignore
from .pipeline.presence import predict_presence, MODEL_LOADED
from .services.policy_loader import load_policies
from .services import policy_loader


app = FastAPI(
    title="BillboardGuard API",
    version="0.1.0",
    description=(
        "Backend for BillboardGuard – AI-powered billboard violation detection "
        "and policy compliance checks."
    ),
)


# CORS: allow Vite dev server and common localhost ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        os.getenv("FRONTEND_URL", "http://localhost:5173"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup_load_policies() -> None:
    # Load legal policies into memory at startup
    load_policies()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _select_analyzer():
    provider = settings.model_provider
    if provider == "mock":
        return analyze_media_mock
    # Placeholders for future providers
    if provider == "ultralytics" and analyze_media_ultra is not None:
        return analyze_media_ultra
    # if provider == "detectron2":
    #     from .pipeline.detectron2 import analyze_media as analyze_media_detectron
    #     return analyze_media_detectron
    return analyze_media_mock


@app.post(
    "/upload",
    response_model=AnalysisResponse,
    summary="Upload media and run billboard detection + compliance checks",
)
async def upload(
    file: UploadFile = File(...),
    metadata: Annotated[str, Form(...)] = None,
):
    try:
        upload_metadata = UploadMetadata.model_validate_json(metadata)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve)) from ve

    file_id = str(uuid.uuid4())
    try:
        saved = await save_upload_file(file_id=file_id, upload=file)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to store file: {exc}") from exc

    media_type: Literal["image", "video"] = "video" if saved.is_video else "image"

    analyzer = _select_analyzer()
    # Step 1: billboard presence classification (binary)
    presence_prob = 1.0
    if settings.presence_enabled:
        try:
            presence_prob = predict_presence(saved.local_path)
        except Exception as exc:  # noqa: BLE001
            presence_prob = 1.0

    # Apply balanced presence thresholds
    if MODEL_LOADED and presence_prob < 0.3:
        empty = DetectionFeatures(
            billboard_count=0,
            estimated_area_sqft=0.0,
            bounding_boxes=[],
            qr_or_license_present=False,
            text_content=[],
        )
        return AnalysisResponse(
            file_id=file_id,
            filename=saved.stored_filename,
            media_type=media_type,
            storage_url=None,
            detections=empty,
            compliance=ComplianceReport(overall_passed=True, checks=[]),
            status="no_billboard",
            message="No billboard detected in this image. Please take a photo of an actual billboard for analysis.",
        )

    if MODEL_LOADED and 0.3 <= presence_prob < 0.5:
        empty = DetectionFeatures(
            billboard_count=0,
            estimated_area_sqft=0.0,
            bounding_boxes=[],
            qr_or_license_present=False,
            text_content=[],
        )
        return AnalysisResponse(
            file_id=file_id,
            filename=saved.stored_filename,
            media_type=media_type,
            storage_url=None,
            detections=empty,
            compliance=ComplianceReport(overall_passed=False, checks=[]),
            status="uncertain",
            message="Uncertain — please confirm if a billboard is present and take a clearer photo.",
        )

    # Step 2: presence >= 0.5 → run detailed detection
    try:
        detections = analyzer(saved.local_path, media_type=media_type)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {exc}") from exc

    # Pre-checks: no billboard → mark invalid
    any_box = len(detections.bounding_boxes) > 0

    if not any_box or detections.billboard_count == 0:
        return AnalysisResponse(
            file_id=file_id,
            filename=saved.stored_filename,
            media_type=media_type,
            storage_url=None,  # do not expose/publicize
            detections=detections,
            compliance=run_compliance_checks(detections, upload_metadata, settings.municipal_area_limit_sqft),
            status="no_billboard",
            message="No billboard detected in this image. Please retake or upload a clearer picture.",
        )

    # Combine presence and box confidence for MVP robustness
    confs = [b.confidence for b in detections.bounding_boxes if b.confidence is not None]
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0
    final_conf = max(avg_conf, presence_prob)
    conf_threshold = settings.detection_conf_threshold
    if final_conf < conf_threshold:
        return AnalysisResponse(
            file_id=file_id,
            filename=saved.stored_filename,
            media_type=media_type,
            storage_url=None,
            detections=detections,
            compliance=run_compliance_checks(detections, upload_metadata, settings.municipal_area_limit_sqft),
            status="uncertain",
            message="Uncertain — needs a clearer photo (low confidence).",
        )

    compliance = run_compliance_checks(
        detections=detections,
        metadata=upload_metadata,
        municipal_area_limit_sqft=settings.municipal_area_limit_sqft,
    )

    return AnalysisResponse(
        file_id=file_id,
        filename=saved.stored_filename,
        media_type=media_type,
        storage_url=saved.public_url,
        detections=detections,
        compliance=compliance,
        status="success",
    )


@app.get("/validate_billboard")
def validate_billboard(
    width: float,
    height: float,
    text: str = "",
    location: str | None = None,
    city: str | None = None,
    qr: bool = False,
):
    size_res = policy_loader.check_size(width, height, city)
    placement_res = policy_loader.check_placement(location, city)
    content_res = policy_loader.check_content(text)
    license_res = policy_loader.check_license(qr)

    overall_passed = all(r.get("status") == "compliant" for r in [size_res, placement_res, content_res, license_res])
    return {
        "overall_passed": overall_passed,
        "checks": [size_res, placement_res, content_res, license_res],
    }


