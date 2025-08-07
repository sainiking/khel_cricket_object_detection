#!/usr/bin/env python3
"""
Quick project setup checker
Verifies that everything is ready for cricket detection training

This script validates:
1. Essential Python files are present
2. Dataset is properly structured
3. Required packages are installed
4. GPU is available for training
"""

# Import required libraries
import os                    # Operating system interface for environment checks
from pathlib import Path     # Modern path handling for file system operations

def check_project_setup():
    """
    Comprehensive check of project setup and dependencies
    Similar to validating data and environment before ML training
    """
    
    # Display header information
    print("Cricket Detection Project Setup Check")
    print("=" * 50)
    
    checks = []    # List to track all check results (True/False)
    
    # Check 1: Verify all required Python files exist
    required_files = [
        "headless_frame_extractor.py",    # Frame extraction from videos
        "train_separate_models.py",       # Separate models training script
        "simple_cricket_detector.py",     # Detection and inference script
        "check_separate_models.py",       # Model capability checker
        "README.md"                       # Documentation
    ]
    
    print("\nChecking project files...")
    for file in required_files:
        if Path(file).exists():                           # Check if file exists
            print(f"  [OK] {file}")
            checks.append(True)                           # Mark check as passed
        else:
            print(f"  [MISSING] {file}")
            checks.append(False)                          # Mark check as failed
    
    # Check 2: Verify Roboflow dataset structure and content
    print("\nChecking dataset...")
    roboflow_dir = Path("roboflow_dataset")               # Expected dataset location
    if roboflow_dir.exists():                             # Check if dataset folder exists
        # Verify required subdirectories exist
        images_dir = roboflow_dir / "train" / "images"    # Training images location
        labels_dir = roboflow_dir / "train" / "labels"    # Training labels location
        
        if images_dir.exists() and labels_dir.exists():  # Both directories must exist
            # Count actual files in directories
            image_count = len(list(images_dir.glob("*.jpg")))     # Count JPEG images
            label_count = len(list(labels_dir.glob("*.txt")))     # Count label files
            
            print(f"  [OK] Roboflow dataset found")
            print(f"  Images: {image_count}")
            print(f"  Labels: {label_count}")
            
            # Verify we have sufficient data for training
            if image_count > 0 and label_count > 0:
                checks.append(True)
            else:
                print(f"  [ERROR] No images or labels found!")
                checks.append(False)
        else:
            print(f"  [ERROR] train/images or train/labels folders missing!")
            checks.append(False)
    else:
        print(f"  [ERROR] roboflow_dataset folder not found!")
        print(f"  Tip: Extract your Roboflow zip file first")
        checks.append(False)
    
    # Check 3: Python packages
    print("\nChecking required packages...")
    required_packages = [
        ("ultralytics", "YOLOv8"),
        ("sklearn", "Data splitting"),
        ("cv2", "Video processing"),
        ("yaml", "Configuration files")
    ]
    
    for package, description in required_packages:
        try:
            if package == "cv2":
                import cv2
            elif package == "sklearn":
                from sklearn.model_selection import train_test_split
            else:
                __import__(package)
            print(f"  [OK] {package} - {description}")
            checks.append(True)
        except ImportError:
            print(f"  [MISSING] {package} - {description}")
            print(f"     Install with: pip install {package}")
            checks.append(False)
    
    # Check 4: GPU availability (optional)
    print("\nChecking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  [OK] GPU available: {gpu_name}")
            checks.append(True)
        else:
            print(f"  [WARNING] No GPU detected (will use CPU - slower)")
            checks.append(True)  # Not critical
    except ImportError:
        print(f"  [WARNING] PyTorch not found (GPU check skipped)")
        checks.append(True)  # Not critical
    
    # Summary
    print("\n" + "=" * 50)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print("ALL CHECKS PASSED!")
        print("Your project is ready for training!")
        print("\nNext steps:")
        print("   1. python3 check_separate_models.py     - Check model status")
        print("   2. python3 train_separate_models.py     - Train separate models")
        print("   3. python3 simple_cricket_detector.py   - Run detection")
    else:
        print(f"WARNING: {passed}/{total} checks passed")
        print("Please fix the issues above before training")
        
        if not Path("roboflow_dataset").exists():
            print("\nQuick fix:")
            print("   1. Download your cricket dataset from Roboflow")
            print("   2. Extract the zip file in this folder")
            print("   3. Unzip the file here, by running: unzip <your_dataset>.zip")
            print("   4. Rename the folder to 'roboflow_dataset'")
            print("   5. Dataset structure for cricket objects (ball uses general YOLO):")
            print("      roboflow_dataset/")
            print("      ├── train/")
            print("      │   ├── images/")
            print("      │   └── labels/")
            

    print("\nFor help, check README.md")

if __name__ == "__main__":
    check_project_setup()
