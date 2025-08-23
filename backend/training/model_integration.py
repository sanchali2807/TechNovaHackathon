#!/usr/bin/env python3
"""
Model integration utilities for custom trained billboard detection models
"""
import os
import shutil
from pathlib import Path
from typing import Optional

class ModelIntegrator:
    def __init__(self, backend_path: str = "../app"):
        self.backend_path = Path(backend_path)
        self.models_dir = self.backend_path.parent / "models"
        self.config_path = self.backend_path / "config.py"
        
    def deploy_custom_model(self, trained_model_path: str, model_name: str = "custom_presence") -> bool:
        """Deploy a custom trained model to the backend"""
        
        trained_model = Path(trained_model_path)
        if not trained_model.exists():
            print(f"Trained model not found: {trained_model_path}")
            return False
        
        # Ensure models directory exists
        self.models_dir.mkdir(exist_ok=True)
        
        # Copy model to backend
        dest_path = self.models_dir / f"{model_name}.onnx"
        shutil.copy2(trained_model, dest_path)
        
        print(f"Custom model deployed to: {dest_path}")
        return True
    
    def update_config_for_custom_model(self, model_name: str = "custom_presence"):
        """Update config.py to use custom model"""
        
        config_content = f'''"""
Configuration settings for Billboard Watch India backend
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR.parent / "models"

# Model settings
MODEL_PROVIDER = "onnx"  # Options: "onnx", "mock"
CUSTOM_MODEL_PATH = str(MODELS_DIR / "{model_name}.onnx")

# Detection settings
presence_enabled: bool = True
presence_threshold_low: float = 0.3   # Below this: reject as non-billboard
presence_threshold_high: float = 0.5  # Above this: proceed to detection
detection_confidence_threshold: float = 0.7  # Minimum confidence for billboard detection

# Upload settings
UPLOAD_DIR = BASE_DIR.parent / "backend" / "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Policies
POLICIES_FILE = BASE_DIR.parent / "policies.json"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class Settings:
    presence_enabled = presence_enabled
    presence_threshold_low = presence_threshold_low
    presence_threshold_high = presence_threshold_high
    detection_confidence_threshold = detection_confidence_threshold
    custom_model_path = CUSTOM_MODEL_PATH if Path(CUSTOM_MODEL_PATH).exists() else None

settings = Settings()
'''
        
        with open(self.config_path, 'w') as f:
            f.write(config_content)
        
        print(f"Updated config.py to use custom model: {model_name}")
    
    def update_main_for_custom_model(self):
        """Update main.py to use custom model path"""
        
        main_path = self.backend_path / "main.py"
        
        # Read current main.py
        with open(main_path, 'r') as f:
            content = f.read()
        
        # Update the presence detection call
        old_call = "presence_prob = predict_presence(saved.local_path)"
        new_call = "presence_prob = predict_presence(saved.local_path, settings.custom_model_path)"
        
        if old_call in content:
            content = content.replace(old_call, new_call)
            
            with open(main_path, 'w') as f:
                f.write(content)
            
            print("Updated main.py to use custom model path")
        else:
            print("Warning: Could not find presence detection call in main.py")
    
    def validate_deployment(self) -> bool:
        """Validate that custom model is properly deployed"""
        
        from ..config import settings
        
        if not settings.custom_model_path:
            print("No custom model path configured")
            return False
        
        if not Path(settings.custom_model_path).exists():
            print(f"Custom model file not found: {settings.custom_model_path}")
            return False
        
        # Test model loading
        try:
            import onnxruntime
            session = onnxruntime.InferenceSession(settings.custom_model_path)
            print(f"Custom model loaded successfully: {settings.custom_model_path}")
            return True
        except Exception as e:
            print(f"Failed to load custom model: {e}")
            return False

def deploy_trained_model(model_path: str, model_name: str = "custom_presence"):
    """Complete deployment of a trained model"""
    
    integrator = ModelIntegrator()
    
    print("Deploying custom trained model...")
    
    # Deploy model file
    if not integrator.deploy_custom_model(model_path, model_name):
        return False
    
    # Update configuration
    integrator.update_config_for_custom_model(model_name)
    
    # Update main.py
    integrator.update_main_for_custom_model()
    
    # Validate deployment
    if integrator.validate_deployment():
        print("✅ Custom model deployed successfully!")
        print("Restart the backend server to use the new model.")
        return True
    else:
        print("❌ Model deployment validation failed")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python model_integration.py <path_to_trained_model.onnx>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    deploy_trained_model(model_path)
