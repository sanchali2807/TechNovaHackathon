from __future__ import annotations

from math import sqrt

from ..models.schemas import ComplianceCheck, ComplianceReport, DetectionFeatures, UploadMetadata
from . import policy_loader


MODEL_POLICY_2016_REFS = [
    "Model Outdoor Advertising Policy 2016 - Section 3 (Size and Placement)",
    "Model Outdoor Advertising Policy 2016 - Section 5 (Content Standards)",
]


def _check_size_limit(detections: DetectionFeatures, city: str | None) -> ComplianceCheck:
    if detections.billboard_count == 0:
        return ComplianceCheck(
            id="size_limit",
            type="size",
            name="Size limit compliance",
            status="compliant",
            passed=True,
            details="No billboards detected",
            policy_reference="Model Outdoor Advertising Policy 2016 - Size Rules",
            references=["Local Municipal Bylaws (area limits)"] + MODEL_POLICY_2016_REFS,
        )

    area = float(detections.estimated_area_sqft or 0.0)
    side = sqrt(area) if area > 0 else 0.0
    res = policy_loader.check_size(side, side, city)
    status = res.get("status", "compliant")
    passed = status == "compliant"
    return ComplianceCheck(
        id="size_limit",
        type="size",
        name="Size limit compliance",
        status=status,
        passed=passed,
        details=res.get("details", ""),
        policy_reference=res.get("policy_reference"),
        references=["Local Municipal Bylaws (area limits)"] + MODEL_POLICY_2016_REFS,
    )


def _check_placement(metadata: UploadMetadata) -> ComplianceCheck:
    city = (metadata.location.city if metadata.location else None) if metadata else None
    location_dict = {
        "distance_to_junction_m": metadata.distance_to_junction_m,
        "distance_to_traffic_light_m": metadata.distance_to_traffic_light_m,
        "is_restricted_zone": metadata.is_restricted_zone,
    }
    res = policy_loader.check_placement(location_dict, city)
    status = res.get("status", "compliant")
    passed = status == "compliant"
    return ComplianceCheck(
        id="placement",
        type="placement",
        name="Placement distance compliance",
        status=status,
        passed=passed,
        details=res.get("details", ""),
        policy_reference=res.get("policy_reference"),
        references=[
            "Typical city policies: min 50–100 m from junctions/signals",
        ]
        + MODEL_POLICY_2016_REFS,
    )


def _check_content(detections: DetectionFeatures) -> ComplianceCheck:
    text = " ".join(detections.text_content or []) if detections.text_content else ""
    res = policy_loader.check_content(text)
    status = res.get("status", "compliant")
    passed = status == "compliant"
    return ComplianceCheck(
        id="content",
        type="content",
        name="Content moderation compliance",
        status=status,
        passed=passed,
        details=res.get("details", ""),
        policy_reference=res.get("policy_reference"),
        references=[
            "IPC/Social decency norms (city policies reference)",
        ]
        + MODEL_POLICY_2016_REFS,
    )


def _check_license_qr(detections: DetectionFeatures) -> ComplianceCheck:
    res = policy_loader.check_license(detections.qr_or_license_present)
    status = res.get("status", "compliant")
    passed = status == "compliant"
    return ComplianceCheck(
        id="license_qr",
        type="license",
        name="License/QR presence compliance",
        status=status,
        passed=passed,
        details=res.get("details", ""),
        policy_reference=res.get("policy_reference"),
        references=["City-specific mandates for QR/license display"] + MODEL_POLICY_2016_REFS,
    )


def run_compliance_checks(
    detections: DetectionFeatures,
    metadata: UploadMetadata,
    municipal_area_limit_sqft: float,
) -> ComplianceReport:
    city = metadata.location.city if metadata and metadata.location else None
    checks = [
        _check_size_limit(detections, city),
        _check_placement(metadata),
        _check_content(detections),
        _check_license_qr(detections),
    ]
    overall_passed = all(c.passed for c in checks)
    return ComplianceReport(overall_passed=overall_passed, checks=checks)


