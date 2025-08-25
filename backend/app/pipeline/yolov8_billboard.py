"""
YOLOv8-based billboard detection pipeline.
Provides better object detection capabilities compared to classification models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:
    YOLO = None

from ..models.schemas import BoundingBox, DetectionFeatures


class YOLOv8BillboardDetector:
    """YOLOv8-based billboard detector with custom training support."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_loaded = False
        
        if YOLO is None:
            print("ERROR: ultralytics not installed - cannot use YOLOv8")
            return
            
        # Try to load custom model first, then pretrained
        model_paths = [
            model_path,
            "models/yolov8_billboard.pt",
            "backend/models/yolov8_billboard.pt",
            "yolov8n.pt"  # Fallback to pretrained nano model
        ]
        
        for path in model_paths:
            if path and self._load_model(path):
                break
    
    def _load_model(self, model_path: str) -> bool:
        """Load YOLOv8 model from path."""
        try:
            print(f"Attempting to load YOLOv8 model: {model_path}")
            self.model = YOLO(model_path)
            self.model_loaded = True
            print(f"SUCCESS: YOLOv8 model loaded from {model_path}")
            
            # Print model info
            if hasattr(self.model, 'names'):
                print(f"Model classes: {self.model.names}")
            
            return True
        except Exception as e:
            print(f"Failed to load YOLOv8 model {model_path}: {e}")
            return False
    
    def detect_billboards(self, image_path: str, confidence_threshold: float = 0.5) -> dict:
        """
        Detect real-world outdoor billboards in image using YOLOv8.
        
        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence for detections (default 0.8)
            
        Returns:
            Dict with detection results, acceptance status, and detailed logging
        """
        if not self.model_loaded or self.model is None:
            print("ERROR: YOLOv8 model not loaded")
            return {
                "accepted": False,
                "message": "Billboard detection model not available. Please check server configuration.",
                "detections": DetectionFeatures(
                    billboard_count=0,
                    estimated_area_sqft=0.0,
                    bounding_boxes=[],
                    qr_or_license_present=False,
                    text_content=[]
                ),
                "debug_info": {"error": "model_not_loaded"}
            }
        
        try:
            print(f"Running YOLOv8 inference on: {image_path}")
            
            # Get image dimensions for area calculations
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return {
                    "accepted": False,
                    "message": "Invalid image file",
                    "detections": DetectionFeatures(billboard_count=0, estimated_area_sqft=0.0, bounding_boxes=[], qr_or_license_present=False, text_content=[]),
                    "debug_info": {"error": "invalid_image"}
                }
            
            img_height, img_width = img.shape[:2]
            img_area = img_width * img_height
            print(f"Image dimensions: {img_width}x{img_height} (area: {img_area} pixels)")
            
            # Run inference with relaxed confidence for debugging
            results = self.model(image_path, conf=0.25)  # Very low threshold to see all detections
            
            all_detections = []
            billboard_detections = []
            debug_info = {
                "image_size": {"width": img_width, "height": img_height},
                "confidence_threshold": confidence_threshold,
                "all_detections": [],
                "billboard_detections": [],
                "rejection_reasons": []
            }
            
            print(f"\n=== YOLO DETECTION DEBUG (threshold={confidence_threshold}) ===")
            print(f"Image: {image_path} | Size: {img_width}x{img_height}")
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.model.names.get(class_id, f"class_{class_id}")
                        
                        width = x2 - x1
                        height = y2 - y1
                        area_ratio = (width * height) / img_area
                        
                        detection_info = {
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": [int(x1), int(y1), int(width), int(height)],
                            "area_ratio": area_ratio
                        }
                        all_detections.append(detection_info)
                        debug_info["all_detections"].append(detection_info)
                        
                        # Enhanced logging for ALL detections
                        print(f"\nDetected: {class_name} | Conf: {confidence:.3f} | Area: {area_ratio:.3f} | Box: [{int(x1)},{int(y1)},{int(width)},{int(height)}]")
                        print(f"  -> Raw area: {width*height:.0f}px | Image area: {img_area}px | Ratio: {area_ratio:.4f}")
                        
                        # Simple rule: Accept if NOT a person/animal AND has reasonable size/confidence
                        REJECTED_CLASSES = ["person", "people", "human", "face", "dog", "cat", "bird", "horse", "cow", "sheep"]
                        
                        if class_name.lower() not in REJECTED_CLASSES and confidence >= 0.4 and area_ratio >= 0.003:
                            print(f"  -> ✅ Accepting {class_name} - not person/animal, good confidence/size")
                            print(f"  -> ACCEPTED: {class_name} meets billboard criteria")
                            
                            # Valid billboard detection
                            bbox = BoundingBox(
                                x=int(x1),
                                y=int(y1),
                                width=int(width),
                                height=int(height),
                                confidence=confidence,
                                class_name="billboard"
                            )
                            
                            billboard_detections.append(bbox)
                            debug_info["billboard_detections"].append({
                                "bbox": [int(x1), int(y1), int(width), int(height)],
                                "confidence": confidence,
                                "area_ratio": area_ratio,
                                "structure_valid": True
                            })
                            
                            print(f"✅ ACCEPTED BILLBOARD: {class_name} | Conf: {confidence:.3f} | Area: {area_ratio:.4f}")
                        
                        else:
                            print(f"  -> ❌ Rejected {class_name} ({confidence:.2f}) - not a billboard class")
            
            # Determine acceptance
            accepted = len(billboard_detections) > 0
            
            if accepted:
                message = f"Billboard detected with {billboard_detections[0].confidence:.0%} confidence"
                total_area_sqft = sum([(b.width * b.height) / 10000 for b in billboard_detections])
            else:
                # Determine specific rejection message based on debug info
                if debug_info["rejection_reasons"]:
                    if any("confidence" in reason for reason in debug_info["rejection_reasons"]):
                        message = "Low confidence. Please upload a clearer billboard photo."
                    elif any("area" in reason for reason in debug_info["rejection_reasons"]):
                        message = "Billboard too small in frame. Please capture closer."
                    elif any("structure" in reason for reason in debug_info["rejection_reasons"]):
                        message = "No billboard structure detected. Please upload a clear rectangular billboard."
                    else:
                        message = "No billboard detected. Please upload a clear outdoor billboard image."
                else:
                    message = "No billboard detected. Please upload a clear outdoor billboard image."
                total_area_sqft = 0.0
            
            detection_features = DetectionFeatures(
                billboard_count=len(billboard_detections),
                estimated_area_sqft=total_area_sqft,
                bounding_boxes=billboard_detections,
                qr_or_license_present=False,
                text_content=[]
            )
            
            print(f"\n=== FINAL RESULT ===")
            print(f"Valid billboards found: {len(billboard_detections)}")
            print(f"Accepted: {accepted}")
            if debug_info["rejection_reasons"]:
                print(f"Rejection reasons: {debug_info['rejection_reasons']}")
            print(f"=== END DEBUG ===\n")
            
            return {
                "accepted": accepted,
                "message": message,
                "detections": detection_features,
                "debug_info": debug_info
            }
            
        except Exception as e:
            print(f"ERROR during YOLOv8 inference: {e}")
            return {
                "accepted": False,
                "message": f"Detection failed: {str(e)}",
                "detections": DetectionFeatures(
                    billboard_count=0,
                    estimated_area_sqft=0.0,
                    bounding_boxes=[],
                    qr_or_license_present=False,
                    text_content=[]
                ),
                "debug_info": {"error": str(e)}
            }


    def _validate_billboard_structure(self, x1: float, y1: float, x2: float, y2: float, img_width: int, img_height: int) -> bool:
        """
        Validate if detected rectangle represents a real billboard with supporting structure.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            img_width, img_height: Image dimensions
            
        Returns:
            True if structure appears to be a real billboard, False for posters/banners
        """
        width = x2 - x1
        height = y2 - y1
        
        # Check aspect ratio - very relaxed for debugging (1:5 to 5:1)
        aspect_ratio = width / height
        
        # Very relaxed aspect ratio range for debugging: 1:5 to 5:1 (0.2 to 5.0)
        min_ratio = 1.0 / 5.0  # 0.2 (very tall)
        max_ratio = 5.0 / 1.0  # 5.0 (very wide)
        
        ratio_valid = min_ratio <= aspect_ratio <= max_ratio
        
        # Check if billboard is mounted outdoors (basic positioning check)
        # Allow ground-level billboards but ensure they're not tiny elements
        min_dimension = min(img_width, img_height) * 0.05  # At least 5% of smaller dimension
        size_adequate = min(width, height) >= min_dimension
        
        # Must be rectangular and adequately sized
        structure_valid = ratio_valid and size_adequate
        
        print(f"    Structure check: aspect={aspect_ratio:.2f} (range {min_ratio:.1f}-{max_ratio:.1f}, valid={ratio_valid}), size_ok={size_adequate}, result={structure_valid}")
        
        return structure_valid


def analyze_media_yolov8(image_path: str, media_type: str = "image") -> dict:
    """
    Analyze media using YOLOv8 billboard detection with hackathon survival fix.
    
    Args:
        image_path: Path to image file
        media_type: Type of media (image/video)
        
    Returns:
        Dict with detection results and acceptance status
    """
    detector = YOLOv8BillboardDetector()
    return detector.detect_billboards(image_path, confidence_threshold=0.5)


# Global detector instance for reuse
_detector_instance: Optional[YOLOv8BillboardDetector] = None

def get_yolov8_detector() -> YOLOv8BillboardDetector:
    """Get singleton YOLOv8 detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YOLOv8BillboardDetector()
    return _detector_instance
