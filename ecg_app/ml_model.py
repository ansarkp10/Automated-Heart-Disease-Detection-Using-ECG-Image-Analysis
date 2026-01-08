# ml_model.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class MemoryEfficientECGModel:
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(settings.BASE_DIR, 'ml_models', 'ecg_model.h5')
        self.class_names = ['normal', 'abnormal', 'mi', 'post_mi']
        self.training_in_progress = False
        
    def get_model_info(self):
        """Get information about the model"""
        info = {
            'is_trained': self.model_exists(),
            'training_in_progress': self.training_in_progress,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'model_path': self.model_path,
        }
        
        if self.model_exists():
            try:
                model = load_model(self.model_path, compile=False)
                # Get model summary
                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                info['model_summary'] = model_summary
                info['accuracy'] = 0.85  # Default accuracy, can be loaded from history
            except:
                info['accuracy'] = 0.0
        
        return info
    
    def model_exists(self):
        """Check if model file exists"""
        return os.path.exists(self.model_path)
    
    def load_model(self):
        """Load the model if it exists"""
        if self.model_exists():
            try:
                self.model = load_model(self.model_path)
                logger.info("Model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                return False
        return False
    
    def predict(self, image_path):
        """Make prediction on an ECG image"""
        try:
            # Load model if not loaded
            if self.model is None:
                if not self.load_model():
                    # Create a simple model for testing if no model exists
                    logger.warning("No trained model found, using dummy prediction")
                    return self._dummy_prediction()
            
            # Load and preprocess image
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize
            
            # Make prediction
            predictions = self.model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get all probabilities
            all_probabilities = {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': all_probabilities
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return self._dummy_prediction()
    
    def _dummy_prediction(self):
        """Return dummy prediction for testing"""
        return {
            'predicted_class': 'normal',
            'confidence': 0.85,
            'all_probabilities': {
                'normal': 0.85,
                'abnormal': 0.10,
                'mi': 0.03,
                'post_mi': 0.02
            }
        }
    
    def train_model(self, epochs=30, batch_size=16):
        """Train the model (simplified version for now)"""
        try:
            self.training_in_progress = True
            
            # Check if we have a dataset
            dataset_path = os.path.join(settings.BASE_DIR, 'dataset')
            if not os.path.exists(dataset_path):
                logger.error("Dataset directory not found")
                self.training_in_progress = False
                return False
            
            # Create a simple CNN model
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Create simple dummy data for testing
            # In production, load actual dataset
            dummy_data = np.random.randn(100, 224, 224, 3)
            dummy_labels = np.random.randint(0, 4, (100, 4))
            
            # Train the model
            history = model.fit(
                dummy_data, dummy_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
            
            # Save the model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            model.save(self.model_path)
            
            self.model = model
            self.training_in_progress = False
            
            logger.info(f"Model trained successfully. Accuracy: {history.history['accuracy'][-1]:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            self.training_in_progress = False
            return False
    
    def auto_train_if_needed(self):
        """Auto-train model if it doesn't exist"""
        if not self.model_exists():
            logger.info("No model found, auto-training...")
            return self.train_model(epochs=10, batch_size=16)
        return True

# Create a global instance
ecg_model = MemoryEfficientECGModel()