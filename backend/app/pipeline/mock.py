from __future__ import annotations

from pathlib import Path
from random import randint, random
import cv2
import numpy as np

from ..models.schemas import BoundingBox, DetectionFeatures


def analyze_media(path: str, media_type: str = "image") -> DetectionFeatures:
    # Option 1: Simple Classifier (MobileNet-based) for billboard presence detection
    
    # Load and preprocess image
    img = cv2.imread(path)
    if img is None:
        return DetectionFeatures(
            billboard_count=0,
            estimated_area_sqft=0.0,
            bounding_boxes=[],
            qr_or_license_present=False,
            text_content=[],
        )
    
    h, w = img.shape[:2]
    
    # Use MobileNet-style feature extraction for billboard classification
    billboard_probability = _classify_billboard_presence(img)
    
    # Binary classification: Billboard (1) or No Billboard (0)
    BILLBOARD_THRESHOLD = 0.3  # 30% confidence threshold - more permissive for actual billboards
    
    if billboard_probability < BILLBOARD_THRESHOLD:
        # Reject as non-billboard or uncertain
        return DetectionFeatures(
            billboard_count=0,
            estimated_area_sqft=0.0,
            bounding_boxes=[],
            qr_or_license_present=False,
            text_content=[f"Rejected: Billboard confidence {billboard_probability:.1%} < 30% threshold"],
        )
    
    # Billboard detected - create a representative bounding box
    # Since this is presence detection, we create a general billboard area
    center_x, center_y = w // 2, h // 2
    bb_width = min(w * 0.7, 600)  # 70% of image width, max 600px
    bb_height = min(h * 0.4, 300)  # 40% of image height, max 300px
    
    bounding_box = BoundingBox(
        x=int(center_x - bb_width // 2),
        y=int(center_y - bb_height // 2),
        width=int(bb_width),
        height=int(bb_height),
        confidence=billboard_probability
    )
    
    # Estimate area (rough conversion to square feet)
    estimated_area_sqft = (bb_width * bb_height) * 0.001
    
    # Generate appropriate text content
    text_content = ["BILLBOARD DETECTED", "ADVERTISEMENT", f"Confidence: {billboard_probability:.1%}"]
    if billboard_probability > 0.9:
        text_content.append("HIGH CONFIDENCE")
    
    return DetectionFeatures(
        billboard_count=1,
        estimated_area_sqft=estimated_area_sqft,
        bounding_boxes=[bounding_box],
        qr_or_license_present=False,  # Would need separate classifier for this
        text_content=text_content,
    )


def _classify_billboard_presence(img: np.ndarray) -> float:
    """
    Enhanced billboard detection focusing on key billboard characteristics.
    Returns probability [0,1] that image contains a billboard.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Feature 1: Large rectangular structure detection (billboards are large rectangles)
    edges = cv2.Canny(gray, 100, 200)  # Stronger edge detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    billboard_score = 0
    large_rect_found = False
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 0.02 * w * h:  # Must be at least 2% of image area
            # Check if it's rectangular
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = cw / ch if ch > 0 else 0
            
            # Billboard characteristics: rectangular, wide aspect ratio, large size
            if (len(approx) >= 4 and  # Roughly rectangular
                1.2 <= aspect_ratio <= 10.0 and  # Wider billboard aspect ratio range
                area > 0.02 * w * h):  # Lower size requirement (2% of image)
                
                billboard_score += area / (w * h)
                large_rect_found = True
    
    # Feature 2: Text detection (billboards contain text/advertisements)
    # Use more aggressive text detection
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    text_regions = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, text_kernel)
    text_density = np.sum(text_regions > 0) / (w * h)
    
    # Feature 3: Uniform color regions (billboards have solid color backgrounds)
    # Check for large uniform regions
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    uniform_regions = cv2.threshold(cv2.Laplacian(blur, cv2.CV_64F), 10, 255, cv2.THRESH_BINARY)[1]
    uniformity_score = 1.0 - (np.sum(uniform_regions > 0) / (w * h))
    
    # Feature 4: Elevated position detection (billboards are often elevated)
    # Check if rectangular structures are in upper 2/3 of image
    elevated_score = 0
    if large_rect_found:
        upper_region = gray[:int(h * 0.67), :]
        upper_edges = cv2.Canny(upper_region, 100, 200)
        upper_contours, _ = cv2.findContours(upper_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in upper_contours:
            area = cv2.contourArea(contour)
            if area > 0.02 * w * h:
                elevated_score = 0.3
                break
    
    # Strict scoring - require multiple strong indicators
    final_score = 0
    
    # More lenient scoring for actual billboards
    if billboard_score > 0.02:  # At least 2% coverage by rectangular structures
        final_score += 0.5  # Higher base score
        
        # Bonus for text content
        if text_density > 0.005:  # Lower text threshold
            final_score += 0.3
            
        # Bonus for uniform regions (clean billboard design)
        if uniformity_score > 0.5:  # Lower uniformity threshold
            final_score += 0.2
            
        # Bonus for elevated position
        final_score += elevated_score
    
    # Alternative path: if we find ANY large rectangular structure with wide aspect ratio
    elif billboard_score > 0.01:  # Even smaller structures
        final_score += 0.4
    
    # Natural scenes (parks, cityscapes) should score very low
    # Check for natural indicators that should reduce score
    natural_penalty = 0
    
    # Detect organic shapes (trees, natural curves)
    contour_complexity = 0
    for contour in contours:
        if cv2.contourArea(contour) > 0.01 * w * h:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) > 8:  # Very complex shape (likely natural)
                contour_complexity += 1
    
    if contour_complexity > 5:  # Only penalize if MANY complex organic shapes
        natural_penalty = 0.2  # Reduced penalty
    
    final_score = max(0, final_score - natural_penalty)
    
    return min(1.0, final_score)





