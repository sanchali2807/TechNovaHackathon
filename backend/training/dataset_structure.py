#!/usr/bin/env python3
"""
Dataset structure creation for billboard detection training
"""
import os
from pathlib import Path

def create_dataset_structure(base_path: str = "billboard_dataset"):
    """Create organized dataset structure for training"""
    
    dataset_path = Path(base_path)
    
    # Create main directories
    directories = [
        "billboards/train",
        "billboards/validation", 
        "billboards/test",
        "non_billboards/train",
        "non_billboards/validation",
        "non_billboards/test",
        "annotations",
        "processed"
    ]
    
    for directory in directories:
        (dataset_path / directory).mkdir(parents=True, exist_ok=True)
        
    # Create README with dataset guidelines
    readme_content = """# Billboard Detection Dataset

## Structure
- billboards/: Images containing billboards
- non_billboards/: Images without billboards (screenshots, indoor, nature, etc.)
- annotations/: JSON files with bounding box annotations
- processed/: Preprocessed images for training

## Data Collection Guidelines

### Billboard Images (Positive Examples)
- Outdoor advertising billboards
- Various angles and distances
- Different lighting conditions
- Include support structures (poles, frames)
- Different sizes and aspect ratios
- Urban and highway billboards

### Non-Billboard Images (Negative Examples)
- Screenshots and UI elements
- Indoor scenes
- Nature landscapes without billboards
- City scenes without visible billboards
- People and vehicles
- Buildings without advertising

## Naming Convention
- billboard_YYYYMMDD_HHMMSS_001.jpg
- non_billboard_type_001.jpg (e.g., non_billboard_screenshot_001.jpg)

## Target Dataset Size
- Minimum: 500 billboard + 500 non-billboard images
- Recommended: 2000+ images total for production use
"""
    
    with open(dataset_path / "README.md", "w") as f:
        f.write(readme_content)
        
    print(f"Dataset structure created at: {dataset_path.absolute()}")
    return dataset_path

if __name__ == "__main__":
    create_dataset_structure()
