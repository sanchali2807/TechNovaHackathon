from __future__ import annotations

from pathlib import Path
from random import randint, random

from ..models.schemas import BoundingBox, DetectionFeatures


def analyze_media(path: str, media_type: str = "image") -> DetectionFeatures:
    # Enhanced mock analyzer with better billboard detection logic
    import cv2
    import numpy as np
    
    # Load and analyze the actual image
    img = cv2.imread(path)
    if img is None:
        return DetectionFeatures(
            billboard_count=0,
            estimated_area_sqft=0.0,
            bounding_boxes=[],
            qr_or_license_present=False,
            text_content=[],
        )
    
    # Basic image analysis to detect billboard-like structures
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check for screenshot-like characteristics (high text density, UI elements)
    # Screenshots typically have high contrast edges and text-like patterns
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (w * h)
    
    # Check for uniform backgrounds (common in screenshots/documents)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_normalized = hist / (w * h)
    dominant_color_ratio = np.max(hist_normalized)
    
    # Simple but effective screenshot detection
    # Screenshots typically have very uniform colors and sharp text edges
    
    # Check for screenshot characteristics:
    # 1. Very high dominant color ratio (uniform backgrounds)
    # 2. High edge density from text and UI elements
    # 3. Specific aspect ratios common in screenshots
    
    screenshot_aspect = w / h if h > 0 else 1
    is_screenshot_aspect = 1.2 < screenshot_aspect < 2.5  # Common screen ratios
    
    # More aggressive screenshot detection
    is_likely_screenshot = (
        dominant_color_ratio > 0.5 or  # Very uniform background
        (edge_density > 0.15 and is_screenshot_aspect) or  # High edges + screen ratio
        edge_density > 0.3  # Very high edge density
    )
    
    if is_likely_screenshot:
        return DetectionFeatures(
            billboard_count=0,
            estimated_area_sqft=0.0,
            bounding_boxes=[],
            qr_or_license_present=False,
            text_content=[],
        )
    
    # Look for large rectangular structures (potential billboards)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    billboard_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.05 * w * h:  # Must be at least 5% of image
            continue
            
        # Get bounding rectangle
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = bw / bh if bh > 0 else 0
        
        # Billboard-like characteristics: wide rectangles (more lenient)
        if aspect_ratio >= 1.2 and area >= 0.05 * w * h:
            billboard_candidates.append({
                'x': x, 'y': y, 'width': bw, 'height': bh,
                'area': area, 'aspect_ratio': aspect_ratio
            })
    
    # If no billboard candidates found, try a more lenient approach for outdoor scenes
    if not billboard_candidates:
        # For outdoor scenes with complex backgrounds, use a different strategy
        # Look for any substantial rectangular areas
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.02 * w * h:
                continue
                
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect_ratio = bw / bh if bh > 0 else 0
            
            # More lenient criteria - billboards can have various aspect ratios
            if aspect_ratio >= 0.8 and area >= 0.02 * w * h:
                billboard_candidates.append({
                    'x': x, 'y': y, 'width': bw, 'height': bh,
                    'area': area, 'aspect_ratio': aspect_ratio
                })
                if len(billboard_candidates) >= 1:  # Found at least one
                    break
    
    # Filter and create bounding boxes
    bounding_boxes = []
    for candidate in billboard_candidates[:2]:  # Max 2 billboards
        bounding_boxes.append(
            BoundingBox(
                x=candidate['x'],
                y=candidate['y'], 
                width=candidate['width'],
                height=candidate['height'],
                confidence=0.8  # Mock confidence
            )
        )
    
    num_billboards = len(bounding_boxes)
    estimated_area_sqft = sum(bb.width * bb.height * 0.001 for bb in bounding_boxes)  # Rough conversion
    
    return DetectionFeatures(
        billboard_count=num_billboards,
        estimated_area_sqft=estimated_area_sqft,
        bounding_boxes=bounding_boxes,
        qr_or_license_present=False,  # Would need OCR for real detection
        text_content=["ADVERTISEMENT"] if num_billboards > 0 else [],
    )




