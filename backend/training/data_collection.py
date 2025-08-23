#!/usr/bin/env python3
"""
Data collection utilities for billboard detection dataset
"""
import os
import requests
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict
import hashlib

class DataCollector:
    def __init__(self, dataset_path: str = "billboard_dataset"):
        self.dataset_path = Path(dataset_path)
        self.annotations = []
        
    def validate_image(self, image_path: str) -> bool:
        """Validate if image is readable and has proper format"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
            h, w = img.shape[:2]
            return h > 100 and w > 100  # Minimum size check
        except:
            return False
    
    def resize_image(self, image_path: str, max_size: int = 1024) -> str:
        """Resize image if too large while maintaining aspect ratio"""
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        if max(h, w) > max_size:
            if h > w:
                new_h = max_size
                new_w = int(w * (max_size / h))
            else:
                new_w = max_size
                new_h = int(h * (max_size / w))
            
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(image_path, img_resized)
            
        return image_path
    
    def add_billboard_image(self, image_path: str, description: str = "", 
                           bounding_boxes: List[Dict] = None) -> bool:
        """Add a billboard image to the dataset"""
        
        if not self.validate_image(image_path):
            print(f"Invalid image: {image_path}")
            return False
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(open(image_path, 'rb').read()).hexdigest()[:8]
        filename = f"billboard_{timestamp}_{file_hash}.jpg"
        
        # Copy to billboard directory
        dest_path = self.dataset_path / "billboards" / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Resize and copy
        self.resize_image(image_path)
        import shutil
        shutil.copy2(image_path, dest_path)
        
        # Add annotation
        annotation = {
            "filename": filename,
            "category": "billboard",
            "description": description,
            "timestamp": timestamp,
            "bounding_boxes": bounding_boxes or []
        }
        self.annotations.append(annotation)
        
        print(f"Added billboard image: {filename}")
        return True
    
    def add_non_billboard_image(self, image_path: str, category: str = "general", 
                               description: str = "") -> bool:
        """Add a non-billboard image to the dataset"""
        
        if not self.validate_image(image_path):
            print(f"Invalid image: {image_path}")
            return False
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(open(image_path, 'rb').read()).hexdigest()[:8]
        filename = f"non_billboard_{category}_{timestamp}_{file_hash}.jpg"
        
        # Copy to non-billboard directory
        dest_path = self.dataset_path / "non_billboards" / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Resize and copy
        self.resize_image(image_path)
        import shutil
        shutil.copy2(image_path, dest_path)
        
        # Add annotation
        annotation = {
            "filename": filename,
            "category": f"non_billboard_{category}",
            "description": description,
            "timestamp": timestamp
        }
        self.annotations.append(annotation)
        
        print(f"Added non-billboard image: {filename}")
        return True
    
    def batch_add_images(self, image_dir: str, is_billboard: bool = True, 
                        category: str = "general") -> int:
        """Batch add images from a directory"""
        
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"Directory not found: {image_dir}")
            return 0
        
        count = 0
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() in image_extensions:
                if is_billboard:
                    if self.add_billboard_image(str(image_path)):
                        count += 1
                else:
                    if self.add_non_billboard_image(str(image_path), category):
                        count += 1
        
        print(f"Added {count} images from {image_dir}")
        return count
    
    def save_annotations(self):
        """Save annotations to JSON file"""
        annotations_path = self.dataset_path / "annotations" / "dataset_annotations.json"
        annotations_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(annotations_path, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        print(f"Saved {len(self.annotations)} annotations to {annotations_path}")
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the current dataset"""
        
        billboard_dir = self.dataset_path / "billboards"
        non_billboard_dir = self.dataset_path / "non_billboards"
        
        billboard_count = len(list(billboard_dir.glob("*.jpg"))) if billboard_dir.exists() else 0
        non_billboard_count = len(list(non_billboard_dir.glob("*.jpg"))) if non_billboard_dir.exists() else 0
        
        stats = {
            "billboard_images": billboard_count,
            "non_billboard_images": non_billboard_count,
            "total_images": billboard_count + non_billboard_count,
            "balance_ratio": billboard_count / max(non_billboard_count, 1)
        }
        
        return stats
    
    def create_sample_data(self):
        """Create sample data structure for demonstration"""
        
        sample_instructions = """
# Sample Data Collection Instructions

## 1. Billboard Images Needed
- Highway billboards (various distances)
- Urban advertising boards
- Bus stop advertisements
- Digital billboards
- Construction site billboards
- Different weather conditions
- Day and night shots

## 2. Non-Billboard Images Needed

### Screenshots (High Priority)
- Email screenshots
- Social media screenshots
- App interface screenshots
- Website screenshots
- Document screenshots

### Indoor Scenes
- Office interiors
- Home interiors
- Shopping mall interiors
- Restaurant interiors

### Nature/Outdoor without Billboards
- Landscapes
- Parks
- Beaches
- Mountains
- City skylines (without visible billboards)

### Other Categories
- People portraits
- Vehicles
- Food
- Animals
- Architecture (buildings without ads)

## 3. Collection Commands

```python
from data_collection import DataCollector

collector = DataCollector()

# Add individual images
collector.add_billboard_image("path/to/billboard.jpg", "Highway billboard")
collector.add_non_billboard_image("path/to/screenshot.jpg", "screenshot", "Email interface")

# Batch add from directories
collector.batch_add_images("billboard_photos/", is_billboard=True)
collector.batch_add_images("screenshots/", is_billboard=False, category="screenshot")
collector.batch_add_images("nature_photos/", is_billboard=False, category="nature")

# Save annotations
collector.save_annotations()

# Check stats
stats = collector.get_dataset_stats()
print(stats)
```
"""
        
        instructions_path = self.dataset_path / "SAMPLE_DATA_INSTRUCTIONS.md"
        with open(instructions_path, 'w') as f:
            f.write(sample_instructions)
        
        print(f"Sample data instructions created: {instructions_path}")

def main():
    """Main data collection interface"""
    
    collector = DataCollector()
    
    # Create sample instructions
    collector.create_sample_data()
    
    # Show current stats
    stats = collector.get_dataset_stats()
    print("\nCurrent Dataset Statistics:")
    print(f"Billboard images: {stats['billboard_images']}")
    print(f"Non-billboard images: {stats['non_billboard_images']}")
    print(f"Total images: {stats['total_images']}")
    print(f"Balance ratio: {stats['balance_ratio']:.2f}")
    
    if stats['total_images'] == 0:
        print("\nDataset is empty. Please add images using the DataCollector class.")
        print("See SAMPLE_DATA_INSTRUCTIONS.md for guidance.")

if __name__ == "__main__":
    main()
