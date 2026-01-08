from django.core.management.base import BaseCommand
from django.conf import settings
from ecg_app.ml_model import ECGModelTrainer
import argparse

class Command(BaseCommand):
    help = 'Train the ECG classification model'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--epochs',
            type=int,
            default=50,
            help='Number of training epochs'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Batch size for training'
        )
        parser.add_argument(
            '--test-only',
            action='store_true',
            help='Only test existing model without training'
        )
    
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting ECG Model Training...'))
        
        trainer = ECGModelTrainer()
        
        if options['test_only']:
            # Test existing model
            try:
                trainer.load_trained_model()
                self.stdout.write(self.style.SUCCESS('Model loaded successfully'))
                
                # Load test data
                X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data()
                
                # Evaluate
                test_results = trainer.evaluate(X_test, y_test)
                
                self.stdout.write(self.style.SUCCESS(
                    f"\nTest Accuracy: {test_results['accuracy']:.4f}"
                ))
                self.stdout.write(self.style.SUCCESS(
                    f"Test Loss: {test_results['loss']:.4f}"
                ))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))
                
        else:
            # Train new model
            try:
                history, test_results = trainer.train(
                    epochs=options['epochs'],
                    batch_size=options['batch_size']
                )
                
                self.stdout.write(self.style.SUCCESS('\n' + '='*50))
                self.stdout.write(self.style.SUCCESS('TRAINING COMPLETED SUCCESSFULLY'))
                self.stdout.write(self.style.SUCCESS('='*50))
                
                self.stdout.write(self.style.SUCCESS(
                    f"\nFinal Test Accuracy: {test_results['accuracy']:.4f}"
                ))
                self.stdout.write(self.style.SUCCESS(
                    f"Final Test Loss: {test_results['loss']:.4f}"
                ))
                
                # Save summary
                summary_path = settings.BASE_DIR / 'training_summary.txt'
                with open(summary_path, 'w') as f:
                    f.write("ECG Model Training Summary\n")
                    f.write("="*50 + "\n")
                    f.write(f"Epochs: {options['epochs']}\n")
                    f.write(f"Batch Size: {options['batch_size']}\n")
                    f.write(f"Test Accuracy: {test_results['accuracy']:.4f}\n")
                    f.write(f"Test Loss: {test_results['loss']:.4f}\n")
                    f.write(f"Test Precision: {test_results['precision']:.4f}\n")
                    f.write(f"Test Recall: {test_results['recall']:.4f}\n")
                    f.write(f"Test F1-Score: {test_results['f1_score']:.4f}\n")
                
                self.stdout.write(self.style.SUCCESS(
                    f"\nTraining summary saved to: {summary_path}"
                ))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error during training: {str(e)}"))