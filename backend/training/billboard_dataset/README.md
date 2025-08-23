# Billboard Detection Dataset

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
