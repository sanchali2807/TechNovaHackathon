"""
Setup script to install YOLOv8 dependencies and download pretrained model.
Run this to prepare the system for YOLOv8 billboard detection.
"""

import subprocess
import sys
from pathlib import Path

def install_ultralytics():
    """Install ultralytics package for YOLOv8."""
    try:
        print("Installing ultralytics package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("✅ Ultralytics installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install ultralytics: {e}")
        return False

def download_yolo_model():
    """Download YOLOv8 pretrained model."""
    try:
        print("Downloading YOLOv8 nano model...")
        from ultralytics import YOLO
        
        # This will download yolov8n.pt if not present
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 model downloaded successfully")
        
        # Test the model
        print("Testing model loading...")
        print(f"Model classes: {model.names}")
        return True
    except Exception as e:
        print(f"❌ Failed to download/test YOLOv8 model: {e}")
        return False

def create_models_directory():
    """Create models directory if it doesn't exist."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"✅ Models directory created: {models_dir.absolute()}")

def main():
    """Main setup function."""
    print("🚀 Setting up YOLOv8 for billboard detection...")
    
    # Create models directory
    create_models_directory()
    
    # Install ultralytics
    if not install_ultralytics():
        print("❌ Setup failed: Could not install ultralytics")
        return False
    
    # Download model
    if not download_yolo_model():
        print("❌ Setup failed: Could not download YOLOv8 model")
        return False
    
    print("\n✅ YOLOv8 setup completed successfully!")
    print("\nNext steps:")
    print("1. The system will use YOLOv8 pretrained model as fallback")
    print("2. For better billboard detection, train a custom model using:")
    print("   python training/train_billboard_model.py --mode yolov8 --data your_dataset")
    print("3. Place custom trained model at: models/yolov8_billboard.pt")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
