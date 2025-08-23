#!/usr/bin/env python3
"""
Billboard detection model training script
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import json
from datetime import datetime

class BillboardDetectionTrainer:
    def __init__(self, dataset_path: str = "billboard_dataset", img_size: tuple = (224, 224)):
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        
    def create_data_generators(self):
        """Create data generators for training and validation"""
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            validation_split=0.2
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            validation_split=0.2
        )
        
        return train_generator, val_generator
    
    def create_model(self):
        """Create MobileNetV2-based model for billboard detection"""
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu', name='feature_dense')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu', name='classifier_dense')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(2, activation='softmax', name='predictions')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        return model, base_model
    
    def compile_model(self, model):
        """Compile model with optimizer and loss"""
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_callbacks(self, model_save_path: str):
        """Create training callbacks"""
        
        callbacks = [
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, fine_tune: bool = True):
        """Train the billboard detection model"""
        
        print("Creating data generators...")
        train_gen, val_gen = self.create_data_generators()
        
        print("Creating model...")
        model, base_model = self.create_model()
        model = self.compile_model(model)
        
        # Create model save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(f"models/billboard_detection_{timestamp}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_save_path = model_dir / "best_model.h5"
        
        print("Starting initial training...")
        callbacks = self.create_callbacks(str(model_save_path))
        
        # Initial training with frozen base
        history1 = model.fit(
            train_gen,
            epochs=20,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        if fine_tune:
            print("Fine-tuning with unfrozen base model...")
            
            # Unfreeze base model for fine-tuning
            base_model.trainable = True
            
            # Use lower learning rate for fine-tuning
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate/10),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Continue training
            history2 = model.fit(
                train_gen,
                epochs=self.epochs,
                initial_epoch=20,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )
        
        print(f"Training completed! Best model saved at: {model_save_path}")
        
        # Save training history
        history_path = model_dir / "training_history.json"
        with open(history_path, 'w') as f:
            history_dict = history1.history
            if fine_tune:
                for key in history_dict:
                    history_dict[key].extend(history2.history[key])
            json.dump(history_dict, f, indent=2)
        
        return model, str(model_save_path)
    
    def export_to_onnx(self, model_path: str, output_path: str = None):
        """Export trained model to ONNX format"""
        
        try:
            import tf2onnx
            
            # Load the trained model
            model = tf.keras.models.load_model(model_path)
            
            if output_path is None:
                output_path = model_path.replace('.h5', '.onnx')
            
            # Convert to ONNX
            spec = (tf.TensorSpec((None, *self.img_size, 3), tf.float32, name="input"),)
            output_path, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
            
            print(f"Model exported to ONNX: {output_path}")
            return output_path
            
        except ImportError:
            print("tf2onnx not installed. Install with: pip install tf2onnx")
            return None

def main():
    """Main training function"""
    
    # Create dataset structure if it doesn't exist
    from dataset_structure import create_dataset_structure
    
    dataset_path = "billboard_dataset"
    if not Path(dataset_path).exists():
        print("Creating dataset structure...")
        create_dataset_structure(dataset_path)
        print(f"Please add images to {dataset_path} before training!")
        return
    
    # Check if dataset has images
    billboard_train = Path(dataset_path) / "billboards"
    non_billboard_train = Path(dataset_path) / "non_billboards"
    
    if not any(billboard_train.iterdir()) or not any(non_billboard_train.iterdir()):
        print("Dataset is empty! Please add images before training.")
        print(f"Add billboard images to: {billboard_train}")
        print(f"Add non-billboard images to: {non_billboard_train}")
        return
    
    # Initialize trainer
    trainer = BillboardDetectionTrainer(dataset_path)
    
    # Train model
    model, model_path = trainer.train(fine_tune=True)
    
    # Export to ONNX
    onnx_path = trainer.export_to_onnx(model_path)
    
    if onnx_path:
        print(f"\nTraining complete!")
        print(f"Keras model: {model_path}")
        print(f"ONNX model: {onnx_path}")
        print(f"\nTo use the trained model, update the model path in backend/app/pipeline/presence.py")

if __name__ == "__main__":
    main()
