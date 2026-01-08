import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import joblib
from django.conf import settings
from .ml_model import ECGModelTrainer

class ECGModelTester:
    def __init__(self):
        self.trainer = ECGModelTrainer()
        self.model = None
        self.label_encoder = None
        self.class_names = []
        
    def load_model(self):
        """Load trained model"""
        try:
            self.model = self.trainer.load_trained_model()
            self.label_encoder = joblib.load(str(settings.ML_CONFIG['LABEL_ENCODER_PATH']))
            self.class_names = list(self.label_encoder.classes_)
            print(f"Model loaded successfully. Classes: {self.class_names}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def test_single_image(self, image_path):
        """Test model on a single image"""
        if not self.model:
            if not self.load_model():
                return None
        
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get class name
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Get all probabilities
            all_probs = {}
            for idx, prob in enumerate(predictions[0]):
                class_name = self.label_encoder.inverse_transform([idx])[0]
                all_probs[class_name] = float(prob)
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'predicted_class_idx': predicted_class_idx
            }
            
        except Exception as e:
            print(f"Error testing image: {str(e)}")
            return None
    
    def batch_test(self, test_images, test_labels):
        """Test model on multiple images"""
        if not self.model:
            if not self.load_model():
                return None
        
        # Preprocess images
        processed_images = []
        for img_path in test_images:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img / 255.0
                processed_images.append(img)
        
        if not processed_images:
            return None
        
        X_test = np.array(processed_images)
        y_test = np.array(test_labels)
        
        # Encode labels
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Make predictions
        predictions = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test_encoded)
        
        # Generate classification report
        report = classification_report(
            y_test_encoded, 
            y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'y_true': y_test_encoded,
            'y_pred': y_pred
        }
    
    def visualize_predictions(self, image_paths, true_labels=None):
        """Visualize predictions for multiple images"""
        if not self.model:
            if not self.load_model():
                return
        
        n_images = len(image_paths)
        n_cols = min(4, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_images > 1 else [axes]
        
        for idx, (img_path, ax) in enumerate(zip(image_paths, axes)):
            if idx >= len(image_paths):
                ax.axis('off')
                continue
            
            result = self.test_single_image(img_path)
            
            if result:
                # Load and display image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                
                # Title with prediction
                title = f"Predicted: {result['predicted_class']}\n"
                title += f"Confidence: {result['confidence']:.2%}"
                
                if true_labels and idx < len(true_labels):
                    title += f"\nTrue: {true_labels[idx]}"
                    # Color code based on correctness
                    if true_labels[idx] == result['predicted_class']:
                        ax.set_title(title, color='green')
                    else:
                        ax.set_title(title, color='red')
                else:
                    ax.set_title(title)
                
                ax.axis('off')
                
                # Add probability distribution as text
                probs_text = "\n".join([f"{cls}: {prob:.2%}" 
                                      for cls, prob in list(result['all_probabilities'].items())[:3]])
                ax.text(0.02, 0.98, probs_text, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, "Error loading image", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        # Hide empty subplots
        for idx in range(len(image_paths), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('prediction_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_performance_report(self, X_test, y_test):
        """Generate comprehensive performance report"""
        if not self.model:
            if not self.load_model():
                return None
        
        # Get batch test results
        batch_results = self.batch_test(X_test, y_test)
        
        if not batch_results:
            return None
        
        # Create detailed report
        report = {
            'overall_accuracy': batch_results['accuracy'],
            'per_class_metrics': batch_results['classification_report'],
            'confusion_matrix': batch_results['confusion_matrix'].tolist(),
            'class_names': self.class_names,
            'num_test_samples': len(X_test)
        }
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(batch_results['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Test Set Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('test_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot class-wise metrics
        metrics = ['precision', 'recall', 'f1-score']
        class_metrics = {metric: [] for metric in metrics}
        
        for class_name in self.class_names:
            for metric in metrics:
                if class_name in batch_results['classification_report']:
                    class_metrics[metric].append(
                        batch_results['classification_report'][class_name][metric]
                    )
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, (metric, values) in enumerate(class_metrics.items()):
            ax.bar(x + idx*width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Class-wise Performance Metrics')
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('class_performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return report

# Global tester instance
tester = ECGModelTester()