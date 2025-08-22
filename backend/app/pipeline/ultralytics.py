from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore

from ..models.schemas import BoundingBox, DetectionFeatures


def _find_billboard_boxes(img: np.ndarray) -> list[BoundingBox]:
    """Detect likely billboard rectangles using contour + polygon approx.

    Heuristics:
    - large quadrilaterals
    - aspect ratio typically wide (2:1 to 6:1)
    - high area relative to image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    # Strengthen edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    # Require sufficiently large candidate region (screen grabs/text columns get filtered)
    min_area = 0.03 * (w * h)  # at least 3% of image area

    candidates: list[tuple[float, BoundingBox]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # approximate polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) < 4:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        aspect = bw / float(bh or 1)
        area_norm = area / float(w * h)
        # Prefer central-upper placements typical of billboards over roads
        y_center = y + bh / 2.0
        y_bias = 1.2 if y_center < h * 0.7 else 0.8

        # Edge density on the border vs inside: a frame-like border boosts score
        border = edges[max(y, 0):min(y+bh, h), max(x, 0):min(x+bw, w)]
        if border.size == 0:
            continue
        inner = border[border.shape[0]//8: -border.shape[0]//8 or None, border.shape[1]//8: -border.shape[1]//8 or None]
        edge_border = float(np.mean(border > 0))
        edge_inner = float(np.mean(inner > 0)) if inner.size else 0.0
        frame_bonus = max(0.0, edge_border - edge_inner)

        # Rectangularity (contour area vs bounding box area)
        rectangularity = area / float(max(bw * bh, 1))

        # Score
        aspect_ok = 1.0 if 2.2 <= aspect <= 8.5 else 0.0
        score = area_norm * 2.0 + aspect_ok + frame_bonus * 1.5 + rectangularity * 0.5
        score *= y_bias
        # Hard filters to avoid false positives like vertical screenshots/text columns
        if (aspect >= 2.0) and (area_norm >= 0.04) and (frame_bonus >= 0.02) and (rectangularity >= 0.6) and score > 1.2:
            # Use score as a proxy for confidence in absence of real model prob.
            conf = float(min(0.99, max(0.5, score)))
            candidates.append((score, BoundingBox(x=int(x), y=int(y), width=int(bw), height=int(bh), confidence=conf)))

    # sort by score desc and return up to 2 boxes
    candidates.sort(key=lambda t: t[0], reverse=True)
    return [box for _, box in candidates[:2]]


def analyze_media(path: str, media_type: str = "image") -> DetectionFeatures:
    p = Path(path)
    img = cv2.imread(str(p))
    if img is None:
        return DetectionFeatures(
            billboard_count=0,
            estimated_area_sqft=0.0,
            bounding_boxes=[],
            qr_or_license_present=False,
            text_content=[],
        )

    boxes = _find_billboard_boxes(img)
    billboard_count = len(boxes)

    # Without calibration we cannot infer true size; leave estimated area minimal.
    est_area_sqft = float(sum(b.width * b.height for b in boxes)) / max(img.shape[0] * img.shape[1], 1) * 400.0 if boxes else 0.0

    # Simple filename tokenization as a weak textual signal (replace with OCR in production)
    tokens = [t for t in p.stem.replace("-", " ").split(" ") if t]

    return DetectionFeatures(
        billboard_count=billboard_count,
        estimated_area_sqft=est_area_sqft,
        bounding_boxes=boxes,
        qr_or_license_present=False,
        text_content=tokens,
    )


