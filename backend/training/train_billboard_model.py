"""
Training script for custom billboard detection model.
Supports both YOLOv8 object detection and MobileNetV2 classification approaches.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v2
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch/Ultralytics not available - install with: pip install torch torchvision ultralytics")


class BillboardDataset(Dataset):
    """Dataset class for billboard classification training."""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]


def prepare_mobilenet_transforms():
    """Prepare transforms for MobileNetV2 training."""
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_mobilenet_classifier(data_dir: str, output_path: str = "models/billboard_mobilenet.pth"):
    """
    Train MobileNetV2 classifier for billboard detection.
    
    Expected directory structure:
    data_dir/
    ├── billboard/     # Images containing billboards
    └── no_billboard/  # Images without billboards
    """
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not available for training")
        return
    
    print("Training MobileNetV2 billboard classifier...")
    
    # Collect data
    billboard_dir = Path(data_dir) / "billboard"
    no_billboard_dir = Path(data_dir) / "no_billboard"
    
    image_paths = []
    labels = []
    
    # Billboard images (label = 1)
    if billboard_dir.exists():
        for img_path in billboard_dir.glob("*.jpg"):
            image_paths.append(str(img_path))
            labels.append(1)
        for img_path in billboard_dir.glob("*.png"):
            image_paths.append(str(img_path))
            labels.append(1)
    
    # No billboard images (label = 0)
    if no_billboard_dir.exists():
        for img_path in no_billboard_dir.glob("*.jpg"):
            image_paths.append(str(img_path))
            labels.append(0)
        for img_path in no_billboard_dir.glob("*.png"):
            image_paths.append(str(img_path))
            labels.append(0)
    
    print(f"Found {len(image_paths)} images ({sum(labels)} billboard, {len(labels) - sum(labels)} no billboard)")
    
    if len(image_paths) < 10:
        print("ERROR: Need at least 10 images for training")
        return
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Prepare transforms
    train_transform, val_transform = prepare_mobilenet_transforms()
    
    # Create datasets
    train_dataset = BillboardDataset(train_paths, train_labels, train_transform)
    val_dataset = BillboardDataset(val_paths, val_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # 2 classes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    num_epochs = 20
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate accuracies
        train_acc = train_correct / len(train_dataset)
        val_acc = val_correct / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"  New best model saved with validation accuracy: {val_acc:.4f}")
        
        scheduler.step()
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_path}")


def train_yolov8_detector(data_dir: str, output_path: str = "models/yolov8_billboard.pt"):
    """
    Train YOLOv8 model for billboard detection.
    
    Expected directory structure:
    data_dir/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
    """
    if not TORCH_AVAILABLE:
        print("ERROR: Ultralytics not available for training")
        return
    
    print("Training YOLOv8 billboard detector...")
    
    data_yaml = Path(data_dir) / "data.yaml"
    if not data_yaml.exists():
        # Create data.yaml if it doesn't exist
        yaml_content = f"""
path: {data_dir}
train: images/train
val: images/val

nc: 1
names: ['billboard']
"""
        with open(data_yaml, 'w') as f:
            f.write(yaml_content)
        print(f"Created data.yaml at {data_yaml}")
    
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # Start with nano pretrained model
    
    # Train the model
    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        name='billboard_detection',
        patience=10,
        save=True,
        plots=True
    )
    
    # Save the trained model
    model.save(output_path)
    print(f"YOLOv8 model saved to: {output_path}")
    
    return results


def create_sample_data_structure():
    """Create sample directory structure for training data."""
    base_dir = Path("training_data")
    
    # For classification (MobileNetV2)
    classification_dir = base_dir / "classification"
    (classification_dir / "billboard").mkdir(parents=True, exist_ok=True)
    (classification_dir / "no_billboard").mkdir(parents=True, exist_ok=True)
    
    # For object detection (YOLOv8)
    detection_dir = base_dir / "detection"
    (detection_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (detection_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (detection_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (detection_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    print("Created sample directory structure:")
    print("training_data/")
    print("├── classification/")
    print("│   ├── billboard/      # Put billboard images here")
    print("│   └── no_billboard/   # Put non-billboard images here")
    print("└── detection/")
    print("    ├── images/")
    print("    │   ├── train/      # Training images")
    print("    │   └── val/        # Validation images")
    print("    └── labels/")
    print("        ├── train/      # YOLO format labels (.txt)")
    print("        └── val/        # YOLO format labels (.txt)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train billboard detection model")
    parser.add_argument("--mode", choices=["mobilenet", "yolov8", "setup"], 
                       default="setup", help="Training mode")
    parser.add_argument("--data", type=str, help="Path to training data directory")
    parser.add_argument("--output", type=str, help="Output model path")
    
    args = parser.parse_args()
    
    if args.mode == "setup":
        create_sample_data_structure()
    elif args.mode == "mobilenet":
        if not args.data:
            print("ERROR: --data required for MobileNet training")
            sys.exit(1)
        output = args.output or "models/billboard_mobilenet.pth"
        train_mobilenet_classifier(args.data, output)
    elif args.mode == "yolov8":
        if not args.data:
            print("ERROR: --data required for YOLOv8 training")
            sys.exit(1)
        output = args.output or "models/yolov8_billboard.pt"
        train_yolov8_detector(args.data, output)
