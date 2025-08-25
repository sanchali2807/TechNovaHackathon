"""
Shape validation service for billboard detection.
Filters out non-billboard structures by analyzing bounding box geometry.
"""

from typing import List
from ..models.schemas import BoundingBox


def is_valid_billboard_shape(bbox: BoundingBox, image_width: int = 1280, image_height: int = 720) -> bool:
    """
    Validate if a bounding box represents a likely billboard structure.
    Loosened constraints to accept tilted or partially obstructed billboards.
    
    Args:
        bbox: Bounding box to validate
        image_width: Width of the source image
        image_height: Height of the source image
    
    Returns:
        True if the shape is likely a billboard, False otherwise
    """
    # Basic size validation - billboard should be substantial (loosened)
    min_width = image_width * 0.08  # At least 8% of image width (reduced from 10%)
    min_height = image_height * 0.04  # At least 4% of image height (reduced from 5%)
    
    if bbox.width < min_width or bbox.height < min_height:
        return False
    
    # Aspect ratio validation - more permissive for tilted billboards
    aspect_ratio = bbox.width / bbox.height
    
    # Expanded valid billboard aspect ratios to handle tilted/obstructed cases
    # Allow wider range: from very vertical to very horizontal
    min_aspect = 0.3  # 1:3.3 (very vertical, reduced from 0.5)
    max_aspect = 12.0  # 12:1 (very wide horizontal, increased from 8.0)
    
    if not (min_aspect <= aspect_ratio <= max_aspect):
        return False
    
    # Position validation - more lenient for edge positioning
    # (allows partially obstructed billboards at edges)
    edge_margin = 0.02  # 2% margin from edges (reduced from 5%)
    
    left_margin = bbox.x / image_width
    right_margin = (bbox.x + bbox.width) / image_width
    top_margin = bbox.y / image_height
    bottom_margin = (bbox.y + bbox.height) / image_height
    
    # Allow billboards closer to edges (for partially visible/obstructed cases)
    if (left_margin < edge_margin or right_margin > (1 - edge_margin) or
        top_margin < edge_margin or bottom_margin > (1 - edge_margin)):
        return False
    
    # Size relative to image - more permissive range
    bbox_area = bbox.width * bbox.height
    image_area = image_width * image_height
    area_ratio = bbox_area / image_area
    
    # Billboard should occupy between 3% and 85% of the image (expanded range)
    if not (0.03 <= area_ratio <= 0.85):
        return False
    
    return True


def filter_billboard_shapes(bboxes: List[BoundingBox], image_width: int = 1280, image_height: int = 720) -> List[BoundingBox]:
    """
    Filter bounding boxes to keep only those that look like billboard structures.
    
    Args:
        bboxes: List of bounding boxes to filter
        image_width: Width of the source image
        image_height: Height of the source image
    
    Returns:
        Filtered list of bounding boxes that pass shape validation
    """
    valid_bboxes = []
    
    for bbox in bboxes:
        if is_valid_billboard_shape(bbox, image_width, image_height):
            valid_bboxes.append(bbox)
    
    return valid_bboxes


def calculate_shape_confidence(bbox: BoundingBox, image_width: int = 1280, image_height: int = 720) -> float:
    """
    Calculate a confidence score based on how billboard-like the shape is.
    
    Args:
        bbox: Bounding box to analyze
        image_width: Width of the source image
        image_height: Height of the source image
    
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not is_valid_billboard_shape(bbox, image_width, image_height):
        return 0.0
    
    confidence = 1.0
    
    # Aspect ratio score - prefer typical billboard ratios
    aspect_ratio = bbox.width / bbox.height
    
    # Ideal aspect ratios for billboards
    ideal_ratios = [2.0, 3.0, 4.0, 6.0]  # Common billboard ratios
    
    # Find closest ideal ratio
    ratio_distances = [abs(aspect_ratio - ideal) for ideal in ideal_ratios]
    min_distance = min(ratio_distances)
    
    # Penalize deviation from ideal ratios
    ratio_score = max(0.5, 1.0 - (min_distance / 4.0))
    confidence *= ratio_score
    
    # Size score - prefer substantial but not overwhelming billboards
    bbox_area = bbox.width * bbox.height
    image_area = image_width * image_height
    area_ratio = bbox_area / image_area
    
    # Ideal area range: 15-40% of image
    if 0.15 <= area_ratio <= 0.4:
        size_score = 1.0
    elif 0.05 <= area_ratio < 0.15:
        size_score = 0.7 + (area_ratio - 0.05) * 3.0  # Scale from 0.7 to 1.0
    elif 0.4 < area_ratio <= 0.8:
        size_score = 1.0 - (area_ratio - 0.4) * 1.25  # Scale from 1.0 to 0.5
    else:
        size_score = 0.3
    
    confidence *= size_score
    
    # Position score - prefer centered objects
    center_x = (bbox.x + bbox.width / 2) / image_width
    center_y = (bbox.y + bbox.height / 2) / image_height
    
    # Distance from image center
    center_distance = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5
    position_score = max(0.7, 1.0 - center_distance)
    confidence *= position_score
    
    return min(1.0, confidence)
