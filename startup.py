#!/usr/bin/env python
"""
Startup script for ECG Heart Disease Detection System
Automatically trains model if needed on startup
"""

import os
import sys
import time

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ecg_project.settings')

import django
django.setup()

from ecg_app.ml_model import ecg_model

def startup_check():
    """Check and initialize model on startup"""
    print("="*60)
    print("ECG HEART DISEASE DETECTION SYSTEM - STARTUP CHECK")
    print("="*60)
    
    # Check model status
    print("\n🔍 Checking model status...")
    model_info = model_manager.get_model_info()
    
    if model_info['model_exists']:
        print(f"✅ Model already exists and is loaded")
        print(f"   Classes: {model_info.get('class_names', [])}")
        if 'accuracy' in model_info:
            print(f"   Accuracy: {model_info['accuracy']:.4f}")
    else:
        print("⚠️  No trained model found")
        
        # Check dataset
        print("\n📁 Checking dataset...")
        dataset_ready, issues = model_manager.check_dataset_ready()
        
        if dataset_ready:
            print("✅ Dataset is ready for training")
            print("\n🚀 Starting automatic model training...")
            
            # Train model automatically
            success = model_manager.auto_train_if_needed()
            
            if success:
                print("\n🎉 Model trained successfully!")
                model_info = model_manager.get_model_info()
                print(f"   Accuracy: {model_info.get('accuracy', 0):.4f}")
            else:
                print("\n❌ Failed to train model automatically")
                print("   Please check your dataset and try manual training")
        else:
            print("\n❌ Dataset not ready for training")
            print("   Issues found:")
            for issue in issues:
                print(f"   - {issue}")
    
    print("\n" + "="*60)
    print("System ready! Start the server with: python manage.py runserver")
    print("="*60)

if __name__ == "__main__":
    startup_check()