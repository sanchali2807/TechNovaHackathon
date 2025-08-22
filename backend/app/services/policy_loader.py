from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..config import settings


POLICIES: Dict[str, Any] | None = None


def load_policies() -> Dict[str, Any]:
    """Load policies JSON from configured path into memory.

    Returns the parsed dictionary and caches it in module global POLICIES.
    """
    global POLICIES
    policies_path: Path = settings.policies_path
    if not policies_path.exists():
        raise FileNotFoundError(f"Policies file not found at {policies_path}")
    with policies_path.open("r", encoding="utf-8") as f:
        POLICIES = json.load(f)
    return POLICIES


def _get_city_rules(city: str | None) -> Dict[str, Any]:
    if POLICIES is None:
        raise RuntimeError("Policies not loaded. Call load_policies() at startup.")

    city_rules: Dict[str, Any] | None = None
    if city:
        municipal = POLICIES.get("municipal_policies", {})
        # try case-insensitive key match
        for key, value in municipal.items():
            if key.lower().startswith(city.lower()):
                city_rules = value
                break
    return city_rules or {}


def _get_national_rules() -> Dict[str, Any]:
    if POLICIES is None:
        raise RuntimeError("Policies not loaded. Call load_policies() at startup.")
    nat = POLICIES.get("national_policies", {})
    return nat.get("model_outdoor_advertising_policy_2016", {})


def check_size(width_ft: float | None, height_ft: float | None, city: str | None) -> Dict[str, str]:
    city_rules = _get_city_rules(city).get("size_rules", {})
    nat_rules = _get_national_rules().get("size_rules", {})

    max_area = city_rules.get("max_area_sqft") or nat_rules.get("max_area_sqft")
    max_height = city_rules.get("max_height_ft") or nat_rules.get("max_height_ft")
    max_width = city_rules.get("max_width_ft") or nat_rules.get("max_width_ft")

    width_val = float(width_ft or 0)
    height_val = float(height_ft or 0)
    area = width_val * height_val

    violations: list[str] = []
    if max_area is not None and area > float(max_area):
        violations.append(f"area {area:.1f} > {float(max_area):.1f} sq ft")
    if max_height is not None and height_val > float(max_height):
        violations.append(f"height {height_val:.1f} > {float(max_height):.1f} ft")
    if max_width is not None and width_val > float(max_width):
        violations.append(f"width {width_val:.1f} > {float(max_width):.1f} ft")

    status = "violation" if violations else "compliant"
    details = "; ".join(violations) if violations else "Within size limits"
    policy_reference = "Model Outdoor Advertising Policy 2016 - Size Rules"
    return {"type": "size", "status": status, "details": details, "policy_reference": policy_reference}


def check_placement(location: str | dict[str, Any] | None, city: str | None) -> Dict[str, str]:
    # Support both string-based descriptors and structured dicts
    loc_text = ""
    if isinstance(location, str):
        loc_text = location.lower()
    elif isinstance(location, dict):
        # Flatten dict values to text for keyword checks
        loc_text = " ".join(str(v) for v in location.values()).lower()

    # Fetch keywords from municipal override first, else national
    city_rules = _get_city_rules(city)
    city_keywords = set(k.lower() for k in city_rules.get("placement_rules", []))
    nat_keywords = set(k.lower() for k in _get_national_rules().get("placement_rules", []))
    keywords = city_keywords or nat_keywords

    hits = sorted([kw for kw in keywords if kw in loc_text]) if loc_text else []
    status = "violation" if hits else "compliant"
    details = (
        f"Restricted placement indicators found: {', '.join(hits)}" if hits else "Placement looks acceptable"
    )
    policy_reference = "Model Outdoor Advertising Policy 2016 - Placement Rules"
    return {"type": "placement", "status": status, "details": details, "policy_reference": policy_reference}


def check_content(text: str | None) -> Dict[str, str]:
    nat = _get_national_rules()
    restrictions: list[str] = nat.get("content_restrictions", [])
    text_l = (text or "").lower()

    # Use provided restricted keywords directly
    keywords = set(k.lower() for k in restrictions)
    # If bullets are phrases, split to tokens of interest
    base_tokens = {
        "obscene",
        "vulgar",
        "derogatory",
        "political",
        "religious",
        "tobacco",
        "alcohol",
    }
    tokens = keywords | base_tokens

    hits = sorted({kw for kw in tokens if kw in text_l})
    status = "violation" if hits else "compliant"
    details = (
        f"Prohibited content detected: {', '.join(hits)}" if hits else "No prohibited content detected"
    )
    policy_reference = "Model Outdoor Advertising Policy 2016 - Content Standards"
    return {"type": "content", "status": status, "details": details, "policy_reference": policy_reference}


def check_license(qr_code_detected: bool | None) -> Dict[str, str]:
    status = "compliant" if bool(qr_code_detected) else "violation"
    details = "License number/QR code present" if status == "compliant" else "License/QR missing"
    policy_reference = "Model Outdoor Advertising Policy 2016 - Licensing Rules"
    return {"type": "license", "status": status, "details": details, "policy_reference": policy_reference}


