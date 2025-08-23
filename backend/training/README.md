# Billboard Detection AI Training System

Complete AI training pipeline for custom billboard detection models.

## Quick Start

### 1. Setup Training Environment
```bash
cd backend/training
pip install -r requirements_training.txt
```

### 2. Create Dataset Structure
```bash
python dataset_structure.py
```

### 3. Collect Training Data
```bash
python data_collection.py
```

### 4. Train Custom Model
```bash
python train_model.py
```

### 5. Deploy Trained Model
```bash
python model_integration.py models/billboard_detection_YYYYMMDD_HHMMSS/best_model.onnx
```

## Files Overview

- **`dataset_structure.py`**: Creates organized dataset folders
- **`data_collection.py`**: Utilities for adding/managing training images
- **`train_model.py`**: Complete training pipeline with MobileNetV2
- **`model_integration.py`**: Deploy trained models to backend
- **`requirements_training.txt`**: Training dependencies

## Dataset Requirements

### Minimum Dataset Size
- **500+ billboard images**: Various angles, lighting, sizes
- **500+ non-billboard images**: Screenshots, indoor, nature, etc.

### Recommended Dataset Size
- **2000+ total images** for production-quality model

## Training Process

1. **Data Preprocessing**: Automatic resizing, validation, augmentation
2. **Transfer Learning**: MobileNetV2 base with custom classification head
3. **Two-Stage Training**: 
   - Initial training with frozen base (20 epochs)
   - Fine-tuning with unfrozen base (30 epochs)
4. **Model Export**: Automatic conversion to ONNX format

## Model Architecture

```
Input (224x224x3)
    ↓
MobileNetV2 Base (ImageNet pretrained)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, relu) + Dropout(0.5)
    ↓
Dense(64, relu) + Dropout(0.3)
    ↓
Dense(2, softmax) → [non_billboard, billboard]
```

## Usage Examples

### Adding Training Data
```python
from data_collection import DataCollector

collector = DataCollector()

# Add individual images
collector.add_billboard_image("billboard.jpg", "Highway billboard")
collector.add_non_billboard_image("screenshot.jpg", "screenshot")

# Batch add from directories
collector.batch_add_images("billboard_photos/", is_billboard=True)
collector.batch_add_images("screenshots/", is_billboard=False, category="screenshot")

# Save annotations
collector.save_annotations()
```

### Training Custom Model
```python
from train_model import BillboardDetectionTrainer

trainer = BillboardDetectionTrainer("billboard_dataset")
model, model_path = trainer.train(fine_tune=True)
onnx_path = trainer.export_to_onnx(model_path)
```

### Deploying Model
```python
from model_integration import deploy_trained_model

deploy_trained_model("path/to/trained_model.onnx", "custom_billboard_v1")
```

## Training Tips

1. **Balanced Dataset**: Equal numbers of billboard/non-billboard images
2. **Diverse Data**: Include various lighting, angles, weather conditions
3. **Quality Control**: Validate all images before training
4. **Screenshot Focus**: Include many UI screenshots as negative examples
5. **Iterative Training**: Start small, evaluate, then expand dataset

## Model Performance

Expected performance with good dataset:
- **Accuracy**: 90-95%
- **Precision**: 85-90% (billboard detection)
- **Recall**: 85-90% (billboard detection)
- **Inference Speed**: ~50ms per image

## Integration

After training, the custom model automatically integrates with your existing backend:
- Updates `config.py` with custom model path
- Modifies `main.py` to use custom model
- Maintains backward compatibility with original system

## Troubleshooting

### Common Issues
- **Low accuracy**: Need more diverse training data
- **High false positives**: Add more non-billboard examples
- **Model won't load**: Check ONNX export process
- **Out of memory**: Reduce batch size in training script

### GPU Training
For faster training, install TensorFlow with GPU support:
```bash
pip install tensorflow[and-cuda]
```

## Next Steps

1. Collect initial dataset (start with 100-200 images)
2. Train baseline model
3. Evaluate performance on test images
4. Iteratively improve dataset based on errors
5. Deploy final model to production
